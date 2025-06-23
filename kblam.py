from torch.nn import Module
import os
import numpy as np
import torch
from typing import List
from torch.nn.parallel import DistributedDataParallel
import evaluate


class KBLaM(Module):
    def __init__(self,
                 tokenizer=None,
                 sentence_encoder=None,
                 llm=None,
                 device="cpu",
                 key_adapter=None,
                 value_adapter=None,
                 separate_query_head=True,
                 separate_query_linear=None,
                 kb_layer_frequency=3,
                 kb_scale_factor=None,
                 use_extended_question_and_answer=False,
                 use_data_augmentation=False):
        super().__init__()
        if torch.cuda.is_available():
            print("cuda is available", torch.cuda.is_available())
        self.device = torch.device(device)

        self.tokenizer = tokenizer
        self.sentence_encoder = sentence_encoder
        self.llm = llm.to(device=self.device)
        self.key_adapter = key_adapter.to(device=self.device)
        self.value_adapter = value_adapter.to(device=self.device)
        self.separate_query_linear = separate_query_linear.to(device=self.device)
        self.rouge_score = evaluate.load("rouge")
        self.bert_score = evaluate.load("bertscore")
        self.bert_score_language = 'en'

        if "llama" in self.llm.config.pretrained_model_name_or_path.lower():
            self.format_input = self.format_input_llama
            self.create_label = self.create_label_llama
            self.prune_text = self.prune_text_llama

        # knowledge base config
        self.kb_layer_frequency = kb_layer_frequency
        self.kb_scale_factor = kb_scale_factor
        self.separate_query_head = separate_query_head
        self.use_extended_question_and_answer = use_extended_question_and_answer
        self.use_data_augmentation = use_data_augmentation

        self.CrossEntropyLoss = torch.nn.CrossEntropyLoss()

        self.to(device=self.device)

    def forward(self, batch_data, context_data, test=False):
        if self.sentence_encoder.training:
            self.sentence_encoder.eval()
            for parameter in self.sentence_encoder.parameters():
                parameter.requires_grad = False
        if self.llm.training:
            self.llm.eval()
            for parameter in self.llm.parameters():
                parameter.requires_grad = False

        input_index, attention_mask, true_label = self.tokenization(batch_data=batch_data)
        knowledge_key_embed, knowledge_value_embed = self.retriever(batch_data=batch_data, context_data=context_data)

        input_index = input_index.to(self.device)
        attention_mask = attention_mask.to(self.device)
        true_label = true_label.to(self.device)
        knowledge_key_embed = knowledge_key_embed.to(self.device)
        knowledge_value_embed = knowledge_value_embed.to(self.device)

        if test:
            kwargs = {}
            kwargs['use_kblam'] = False
            kwargs['separate_query_head'] = self.separate_query_head
            kwargs['knowledge_embed'] = (knowledge_key_embed, knowledge_value_embed)
            kwargs['separate_query_linear'] = self.separate_query_linear
            kwargs['kb_layer_frequency'] = self.kb_layer_frequency
            kwargs['kb_scale_factor'] = self.kb_scale_factor

            pred_token_index_NoKB = self.llm.generate(input_ids=input_index,  # Llama
                                                      attention_mask=attention_mask,
                                                      # KBLaM
                                                      kwargs=kwargs,
                                                      # generate
                                                      max_new_tokens=40,
                                                      tokenizer=self.tokenizer)
            kwargs['use_kblam'] = True
            pred_token_index_UsKB = self.llm.generate(input_ids=input_index,  # Llama
                                                      attention_mask=attention_mask,
                                                      # KBLaM
                                                      kwargs=kwargs,
                                                      # generate
                                                      max_new_tokens=40,
                                                      tokenizer=self.tokenizer)

            return pred_token_index_NoKB, pred_token_index_UsKB


        else:
            out = self.llm.forward(input_ids=input_index,  # Llama
                                   attention_mask=attention_mask,
                                   # KBLaM
                                   use_kblam=True,
                                   separate_query_head=self.separate_query_head,
                                   knowledge_embed=(knowledge_key_embed, knowledge_value_embed),
                                   separate_query_linear=self.separate_query_linear,
                                   kb_scale_factor=self.kb_scale_factor,
                                   kb_layer_frequency=self.kb_layer_frequency)
            logits = out["logits"]

            return logits, true_label

    def loss_function(self, logits, true_label):
        batch_size, seq_len, vocab_size = logits.shape
        # pred_label = logits.argmax(axis=2)
        # for batch_index in range(batch_size):
        #     token_index = pred_label[batch_index]
        #     pred_text = self.tokenizer.decode(token_index)
        #     token_index = true_label[batch_index]
        #     token_index = token_index[token_index >= 0]
        #     true_text = self.tokenizer.decode(token_index)

        shift_logits = logits[:, :-1, :].contiguous()  # 移位logits以对应标签
        shift_labels = true_label[:, 1:].contiguous()  # 移位标签

        weights = (shift_labels > 0)
        weights = weights.sum(-1, keepdim=True)
        weights = weights.expand(-1, shift_labels.shape[1])
        weights = weights.contiguous()

        if not isinstance(self.llm, DistributedDataParallel):
            assert vocab_size == self.llm.config.vocab_size
        else:
            assert vocab_size == self.llm.module.config.vocab_size

        shift_logits = shift_logits.view(-1, vocab_size)  # 重塑logits
        shift_labels = shift_labels.view(-1)  # 重塑标签
        weights = weights.view(-1)  # 重塑权重

        shift_labels = shift_labels.to(shift_logits.device)
        loss = self.CrossEntropyLoss(input=shift_logits, target=shift_labels)
        # loss = (loss * weights.max() / weights).mean()
        loss = loss * (weights.max() / weights)
        loss = loss.mean()

        return loss

    def score_metrics(self, predictions, references):
        rouge_score = self.rouge_score.compute(predictions=predictions, references=references)
        bert_score = self.bert_score.compute(predictions=predictions, references=references,
                                             lang=self.bert_score_language)
        for name, value in bert_score.items():
            if isinstance(value, list):
                bert_score[name] = np.mean(value)

        return rouge_score, bert_score

    def tokenization(self, batch_data):
        with torch.autograd.no_grad():
            batch_format_QA = []
            for data in batch_data:
                Q, A = self.select_question_and_answer(data=data)
                format_QA = self.format_input(Q=Q, A=A)
                batch_format_QA.append(format_QA)
            batch_format_QA = self.tokenizer(batch_format_QA, return_tensors="pt", padding=True)
            input_index = batch_format_QA["input_ids"]
            attention_mask = batch_format_QA["attention_mask"]
            true_label = self.create_label(input_ids=input_index, input_strs=batch_format_QA)

        return input_index, attention_mask, true_label

    def retriever(self, batch_data, context_data):
        batch_size = len(batch_data)
        # 1 knowledge token
        key_embed_list, value_embed_list = [], []
        for data in batch_data:
            key_text = data["key_string"]
            key_embed = self.sentence_encoder.forward(sentence=key_text)
            key_embed = self.key_adapter(key_embed)
            key_embed_list.append(key_embed)

            value_text = data["description"]
            value_embed = self.sentence_encoder.forward(sentence=value_text)
            value_embed = self.value_adapter(value_embed)
            value_embed_list.append(value_embed)

        batch_key_embed = torch.stack(key_embed_list)
        batch_value_embed = torch.stack(value_embed_list)

        if len(batch_key_embed.shape) == 2:
            batch_size, embed_dim = batch_key_embed.shape
            batch_size, embed_dim = batch_value_embed.shape
            batch_key_embed = batch_key_embed.unsqueeze(1)
            batch_value_embed = batch_value_embed.unsqueeze(1)

        # 2 context token
        context_key_embed_list, context_value_embed_list = [], []
        for data in context_data:
            key_text = data["key_string"]
            key_embed = self.sentence_encoder.forward(sentence=key_text)
            key_embed = self.key_adapter(key_embed)
            context_key_embed_list.append(key_embed)

            value_text = data["description"]
            value_embed = self.sentence_encoder.forward(sentence=value_text)
            value_embed = self.value_adapter(value_embed)
            context_value_embed_list.append(value_embed)

        context_key_embed = torch.stack(context_key_embed_list)
        context_value_embed = torch.stack(context_value_embed_list)

        context_size, embed_dim = context_key_embed.shape
        context_size, embed_dim = context_value_embed.shape
        context_key_embed = context_key_embed.unsqueeze(0).expand(batch_size, context_size, embed_dim)
        context_value_embed = context_value_embed.unsqueeze(0).expand(batch_size, context_size, embed_dim)

        true_kb_copy = 1
        batch_key_embed = [batch_key_embed] * true_kb_copy + [context_key_embed]
        batch_value_embed = [batch_value_embed] * true_kb_copy + [context_value_embed]
        batch_key_embed = torch.concat(tensors=batch_key_embed, dim=1)
        batch_value_embed = torch.concat(tensors=batch_value_embed, dim=1)

        batch_size, seq_len, embed_dim = batch_key_embed.shape
        seq_len = batch_size // batch_size + context_size

        return batch_key_embed, batch_value_embed

    def select_question_and_answer(self, data):
        Q, A = None, None
        if self.use_extended_question_and_answer:
            Q = data["extended_Q"]
            A = data["extended_A"]

        elif self.use_data_augmentation:
            data = data
            templates = ["What {} does {} have?",
                         "What is the {} of {}?",
                         "Tell me about the {} of {}.",
                         "Can you let me know the {} of {}?",
                         "Can you inform me about the {} of {}?",
                         "Describe the {} of {}.",
                         "What details can you share about the {} of {}?",
                         "What kind of {} does {} have?",
                         "Provide details on the {} of {}.",
                         "What features does the {} of {} include?",
                         "Can you elaborate on the {} of {}?",
                         "How would you describe the {} of {}?",
                         "What can you tell me about the {} characteristics of {}?",
                         "Can you explain the {} of {}?",
                         "What insights can you provide about the {} of {}?",
                         "What should I know about the {} of {}?"]
            dtype = data["description_type"]
            name = data["name"]
            tid = np.random.randint(0, len(templates))
            Q = templates[tid].format(dtype, name)
            A = "I am sorry I cannot find relevant information in the KB."

        else:
            Q = data["Q"]
            A = data["A"]

        assert Q is not None and A is not None
        return Q, A

    def format_input_llama(self, Q: str, A: str):
        text = "<|start_header_id|>user<|end_header_id|> "
        text += Q
        text += "<|eot_id|>"
        text += "<|start_header_id|>assistant<|end_header_id|>"
        text += A
        text += "<|eot_id|>"
        return text

    def create_label_llama(self, input_ids: torch.Tensor, input_strs: List[str]):
        # Not sure this is correct. This method simply masks the <|start_header_id|>user<|end_header_id|>
        # then leaves the rest in the labels
        # Possibly what they want is to mask out the query.
        # To do that swap the index from the tokenizer below from 1 to 2
        answer_indices = torch.argmax(
            (input_ids == self.tokenizer("<|start_header_id|>assistant<|end_header_id|>")["input_ids"][1]).long(),
            -1,
        )
        answer_mask = torch.ones_like(input_ids)
        for b in range(len(input_strs)):
            answer_mask[b, : (answer_indices[b].item() + 2)] = 0
        labels = input_ids * answer_mask + (1 - answer_mask) * (-100)
        return labels

    def prune_text_llama(self, sentence: str) -> str:
        # 原始句子
        sentence = sentence.replace("<|eot_id|>", "")
        # 替换 header_id 的标记
        sentence = sentence.replace("<|start_header_id|>assistant<|end_header_id|>", "")
        sentence = sentence.replace("<|start_header_id|>user<|end_header_id|>", "")
        # 替换文本结束标记
        sentence = sentence.replace("<|end_of_text|>", "")
        return sentence

    def save_attention(self, each_layer_attn_weights, attention_save_dir, attention_save_name):
        for layer_index in range(len(each_layer_attn_weights)):
            attn_weights = each_layer_attn_weights[layer_index]
            # TODO: Make this function injectable
            save_path = os.path.join(attention_save_dir, f"{attention_save_name}_LayerIndex{self.layer_idx}")
            save_path = save_path + ".npy"
            np.save(save_path, attn_weights.to(torch.float32).cpu().detach().numpy())
