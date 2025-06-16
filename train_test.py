import json
import logging
import os
import pathlib
import wandb
import numpy as np
import torch
from transformers import AutoTokenizer
from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme
from accelerate import Accelerator
from torch.nn import Identity

from dataloader import Dataloader
from transformers.models.llama.modeling_llama import LlamaConfig
from llama_kblam import LlamaForCausalLM
from sentence_transformer import SentenceEncoder
from adapter import KeyAdapter, ValueAdapter
from separate_query_projector import SeparateQueryLinear
from kblam import KBLaM
from early_stop import EarlyStopping

device = "cpu"
torch.manual_seed(seed=1)
np.random.seed(seed=1)

torch.cuda.empty_cache()

dataloader = Dataloader()

pretrained_model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
llm_config = LlamaConfig(num_hidden_layers=1,
                         vocab_size=128009 + 1000,
                         pretraining_tp=1,
                         pretrained_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                          trust_remote_code=True,
                                          padding_side='left',
                                          token="")
tokenizer.pad_token = tokenizer.eos_token
llm_config = LlamaConfig(num_hidden_layers=2,
                         vocab_size=len(tokenizer.vocab),
                         pretraining_tp=1,
                         pretrained_model_name_or_path=pretrained_model_name_or_path)

llm = LlamaForCausalLM(config=llm_config)
sentence_encoder = SentenceEncoder(model_name="sentence-transformers/all-mpnet-base-v2",
                                   device=device)
key_adapter = KeyAdapter(in_dim=sentence_encoder.out_dim,
                         out_dim=llm.config.hidden_size)
value_adapter = ValueAdapter(in_dim=sentence_encoder.out_dim,
                             out_dim=llm.config.hidden_size)
separate_query_linear = SeparateQueryLinear(in_dim=llm.config.hidden_size,
                                            out_dim=llm.config.hidden_size,
                                            num_layer=llm.config.num_hidden_layers)
kblam = KBLaM(tokenizer=tokenizer,
              sentence_encoder=sentence_encoder,
              key_adapter=key_adapter,
              value_adapter=value_adapter,
              separate_query_head=True,
              separate_query_linear=separate_query_linear,
              kb_layer_frequency=3,
              llm=llm)

num_epoch = 10
learning_rate = 0.001
eta_min = learning_rate * 0.01

trainable_parameters = list(kblam.key_adapter.parameters()) + list(kblam.value_adapter.parameters())
if kblam.separate_query_head:
    trainable_parameters += list(kblam.separate_query_linear.parameters())

optimizer = torch.optim.AdamW(params=trainable_parameters, lr=learning_rate, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epoch, eta_min=eta_min)
stopper = EarlyStopping()

kblam.sentence_encoder.eval()
kblam.llm.eval()
# for epoch in range(0, num_epoch):
#     # train
#     optimizer.zero_grad()
#     kblam.key_adapter.train()
#     kblam.value_adapter.train()
#     kblam.separate_query_linear.train()
#     for batch_index in range(dataloader.num_train_batch):
#         batch_data, context_data = dataloader.train_dataloader(epoch=epoch, batch_index=batch_index)
#         logits, true_label = kblam.forward(batch_data=batch_data, context_data=context_data)
#         loss = kblam.loss_function(logits=logits, true_label=true_label)
#         loss.backward()
#         optimizer.step()
#         break
#     scheduler.step()
#     dataloader.shuffle_train_data()
#
#     # validation
#     with torch.no_grad():
#         kblam.eval()
#         valid_loss = []
#         for batch_index in range(dataloader.num_valid_batch):
#             batch_data, context_data = dataloader.valid_dataloader(epoch=epoch, batch_index=batch_index)
#             logits, true_label = kblam.forward(batch_data=batch_data, context_data=context_data)
#             loss = kblam.loss_function(logits=logits, true_label=true_label)
#             valid_loss.append(loss.item())
#             break
#         valid_loss = sum(valid_loss) / len(valid_loss)
#
#     stopper.record(now_val_loss=valid_loss, model=kblam)
#     if stopper.is_stop:
#         break
#     else:
#         dataloader.shuffle_train_data()
#         dataloader.shuffle_valid_data()
#         stopper.is_stop = True
#
# if stopper.is_stop:
#     save_best_model_path = 'best+'
#     save_best_model_path += dataloader.dataset_name + '+'
#     save_best_model_path += sentence_encoder.model_name + '+'
#     save_best_model_path += llm_config.pretrained_model_name_or_path + '.pth'
#     save_best_model_path = save_best_model_path.replace('/','_')
#     torch.save(stopper.best_model_parameter_state_dict, save_best_model_path)

# test
with torch.no_grad():
    # kblam.load_state_dict(state_dict=stopper.best_model_parameter_state_dict)
    kblam.eval()
    kblam.llm.generation_config.pad_token_id = tokenizer.pad_token_id
    kblam.llm.generation_config.eos_token_id = tokenizer.eos_token_id

    pred_answer_without_kb_list = []
    pred_answer_with_kb_list = []
    true_answer_list = []

    for batch_index in range(dataloader.num_test_batch):
        batch_data, context_data = dataloader.test_dataloader(epoch=batch_index, batch_index=batch_index)
        pred_token_index_without_kb, pred_token_index_with_kb = kblam.forward(batch_data=batch_data,
                                                                              context_data=context_data,
                                                                              test=True)
        pred_token_index_without_kb = tokenizer.batch_decode(pred_token_index_without_kb, skip_special_tokens=False)
        pred_token_index_with_kb = tokenizer.batch_decode(pred_token_index_with_kb, skip_special_tokens=False)

        for index, data in enumerate(batch_data):
            # 1
            pred_answer_without_kb = pred_token_index_without_kb[index]
            pred_answer_without_kb = kblam.prune_text_llama(sentence=pred_answer_without_kb)
            question = data["Q"]
            pred_answer_without_kb = pred_answer_without_kb.split(question)
            if len(pred_answer_without_kb) > 1:
                pred_answer_without_kb = pred_answer_without_kb[1]
            else:
                pred_answer_without_kb = ""
            pred_answer_without_kb_list.append(pred_answer_without_kb)

            # 2
            pred_answer_with_kb = pred_token_index_without_kb[index]
            pred_answer_with_kb = kblam.prune_text_llama(sentence=pred_answer_with_kb)
            question = data["Q"]
            pred_answer_with_kb = pred_answer_with_kb.split(question)
            if len(pred_answer_with_kb) > 1:
                pred_answer_with_kb = pred_answer_with_kb[1]
            else:
                pred_answer_with_kb = ""
            pred_answer_with_kb_list.append(pred_answer_with_kb)

            # 3
            true_answer = data["A"]
            true_answer_list.append(true_answer)

        break

    import evaluate

    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")

    rogue_score_without_kb = rouge.compute(predictions=pred_answer_without_kb_list, references=true_answer_list)
    rogue_score_with_kb = rouge.compute(predictions=pred_answer_with_kb_list, references=true_answer_list)

    print()
