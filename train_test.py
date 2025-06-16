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
import pandas as pd
from datetime import datetime
import copy
import evaluate

large_language_model_name = "meta-llama/Llama-3.2-1B-Instruct"
# large_language_model_name = "meta-llama/Llama-3.2-3B-Instruct"
# large_language_model_name = "meta-llama/Llama-3.1-8B-Instruct"
# large_language_model_name = "meta-llama/Llama-3.3-70B-Instruct"
# sentence_transformer_name = "sentence-transformers/all-MiniLM-L6-v2"
sentence_transformer_name = "sentence-transformers/all-mpnet-base-v2"
# sentence_transformer_name = "text-embedding-3-large"
# sentence_transformer_name = "ada-embeddings"

debug = True
device = "cuda"
# device = "cpu"
dataset_name = "synthetic.json"
# dataset_name = "enron.json"
tokenizer_padding_side = 'left'
num_epoch = 10
batch_size = 10
stopper_patience = 10
optimizer_learning_rate = 0.001
optimizer_weight_decay = 0.01
scheduler_eta_min = optimizer_learning_rate * 0.01
seed = 1

torch.manual_seed(seed=seed)
np.random.seed(seed=seed)
torch.cuda.empty_cache()
dataloader = Dataloader(dataset_name=dataset_name, batch_size=batch_size, seed=seed)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=large_language_model_name,
                                          trust_remote_code=True,
                                          padding_side='left',
                                          token="")
tokenizer.pad_token = tokenizer.eos_token
llm_config = LlamaConfig(num_hidden_layers=2,
                         vocab_size=len(tokenizer.vocab),
                         pretraining_tp=1,
                         pretrained_model_name_or_path=large_language_model_name)
llm = LlamaForCausalLM(config=llm_config)
sentence_encoder = SentenceEncoder(model_name=sentence_transformer_name,
                                   device=device if device.lower() == "cpu" else None)
key_adapter = KeyAdapter(in_dim=sentence_encoder.out_dim, out_dim=llm.config.hidden_size)
value_adapter = ValueAdapter(in_dim=sentence_encoder.out_dim, out_dim=llm.config.hidden_size)
separate_query_linear = SeparateQueryLinear(in_dim=llm.config.hidden_size, out_dim=llm.config.hidden_size,
                                            num_layer=llm.config.num_hidden_layers)
kblam = KBLaM(device=device,
              tokenizer=tokenizer,
              sentence_encoder=sentence_encoder,
              key_adapter=key_adapter,
              value_adapter=value_adapter,
              separate_query_head=True,
              separate_query_linear=separate_query_linear,
              kb_layer_frequency=3,
              llm=llm)

trainable_parameters = list(kblam.key_adapter.parameters()) + list(kblam.value_adapter.parameters())
if kblam.separate_query_head:
    trainable_parameters += list(kblam.separate_query_linear.parameters())

optimizer = torch.optim.AdamW(params=trainable_parameters,
                              lr=optimizer_learning_rate,
                              weight_decay=optimizer_weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=num_epoch,
                                                       eta_min=scheduler_eta_min)
stopper = EarlyStopping(patience=stopper_patience)

for epoch in range(0, num_epoch):
    if kblam.sentence_encoder.training:
        kblam.sentence_encoder.eval()
    if kblam.llm.training:
        kblam.llm.eval()

    # train
    optimizer.zero_grad()
    kblam.key_adapter.train()
    kblam.value_adapter.train()
    kblam.separate_query_linear.train()
    for batch_index in range(dataloader.num_train_batch):
        batch_data, context_data = dataloader.train_dataloader(epoch=epoch, batch_index=batch_index)
        logits, true_label = kblam.forward(batch_data=batch_data, context_data=context_data)
        loss = kblam.loss_function(logits=logits, true_label=true_label)
        loss.backward()
        optimizer.step()
        if debug: break
    scheduler.step()
    dataloader.shuffle_train_data()

    # validation
    with torch.no_grad():
        kblam.eval()
        valid_loss = []
        for batch_index in range(dataloader.num_valid_batch):
            batch_data, context_data = dataloader.valid_dataloader(epoch=epoch, batch_index=batch_index)
            logits, true_label = kblam.forward(batch_data=batch_data, context_data=context_data)
            loss = kblam.loss_function(logits=logits, true_label=true_label)
            valid_loss.append(loss.item())
            if debug: break
        valid_loss = sum(valid_loss) / len(valid_loss)

    stopper.record(now_val_loss=valid_loss, model=kblam)
    if stopper.is_stop:
        break
    else:
        dataloader.shuffle_train_data()
        dataloader.shuffle_valid_data()
        if debug: stopper.is_stop = True

if stopper.is_stop:
    save_best_model_path = 'best+'
    save_best_model_path += dataloader.dataset_name + '+'
    save_best_model_path += sentence_encoder.model_name + '+'
    save_best_model_path += llm_config.pretrained_model_name_or_path + '.pth'
    save_best_model_path = save_best_model_path.replace('/', '_')
    torch.save(stopper.best_model_parameter_state_dict, save_best_model_path)

# test
with torch.no_grad():
    kblam.load_state_dict(state_dict=stopper.best_model_parameter_state_dict)
    kblam.eval()
    kblam.llm.generation_config.pad_token_id = tokenizer.pad_token_id
    kblam.llm.generation_config.eos_token_id = tokenizer.eos_token_id

    pred_answer_no_kb_list = []
    pred_answer_use_kb_list = []
    true_answer_list = []

    for batch_index in range(dataloader.num_test_batch):
        batch_data, context_data = dataloader.test_dataloader(epoch=batch_index, batch_index=batch_index)
        pred_token_index_no_kb, pred_token_index_use_kb = kblam.forward(batch_data=batch_data,
                                                                        context_data=context_data,
                                                                        test=True)
        pred_token_index_no_kb = tokenizer.batch_decode(pred_token_index_no_kb, skip_special_tokens=False)
        pred_token_index_use_kb = tokenizer.batch_decode(pred_token_index_use_kb, skip_special_tokens=False)

        for index, data in enumerate(batch_data):
            # 1
            pred_answer_no_kb = pred_token_index_no_kb[index]
            pred_answer_no_kb = kblam.prune_text(sentence=pred_answer_no_kb)
            question = data["Q"]
            pred_answer_no_kb = pred_answer_no_kb.split(question)
            if len(pred_answer_no_kb) > 1:
                pred_answer_no_kb = pred_answer_no_kb[1]
            else:
                pred_answer_no_kb = ""
            pred_answer_no_kb_list.append(pred_answer_no_kb)

            # 2
            pred_answer_use_kb = pred_token_index_no_kb[index]
            pred_answer_use_kb = kblam.prune_text(sentence=pred_answer_use_kb)
            question = data["Q"]
            pred_answer_use_kb = pred_answer_use_kb.split(question)
            if len(pred_answer_use_kb) > 1:
                pred_answer_use_kb = pred_answer_use_kb[1]
            else:
                pred_answer_use_kb = ""
            pred_answer_use_kb_list.append(pred_answer_use_kb)

            # 3
            true_answer = data["A"]
            true_answer_list.append(true_answer)

        if debug: break

    rogue_score_no_kb, bert_score_no_kb = kblam.score_metrics(predictions=pred_answer_no_kb_list,
                                                              references=true_answer_list)
    rogue_score_use_kb, bert_score_use_kb = kblam.score_metrics(predictions=pred_answer_use_kb_list,
                                                                references=true_answer_list)

print('test done')


# save result
def get_parameter(model=None, items=None):
    out = {}
    if model and items is None:
        out['str(model)'] = str(model)
        for name, value in model.__dict__.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                out[name] = value
            else:
                out[name] = name
        return out
    elif items and model is None:
        for name, value in items:
            if isinstance(value, (int, float, str, bool, type(None))):
                out[name] = value
            else:
                out[name] = name
        return out
    else:
        return out


kblam_config = get_parameter(model=kblam)
llm_config = get_parameter(model=kblam.llm)
sentence_encoder_config = get_parameter(model=kblam.sentence_encoder)
key_adapter_config = get_parameter(model=kblam.key_adapter)
value_adapter_config = get_parameter(model=kblam.value_adapter)
separate_query_linear_config = get_parameter(model=kblam)
global_config = get_parameter(items=globals().items())

all_global_dict = {}
globals_value = list(globals().items())  # 将 items() 转换为列表
for name, value in globals_value:
    if isinstance(value, dict) and not name.startswith('__'):
        all_global_dict[name] = value

df = pd.DataFrame()
for i, (dict_name, now_dict) in enumerate(all_global_dict.items()):
    name = pd.DataFrame([dict_name], index=[2 * i])
    keys = pd.DataFrame([now_dict.keys()], index=[2 * i + 1])
    values = pd.DataFrame([now_dict.values()], index=[2 * i + 2])
    df = pd.concat([df, name, keys, values], ignore_index=False)
df.to_excel(f"{datetime.now().strftime(('%Y-%m-%d %H-%M-%S'))}.xlsx")
