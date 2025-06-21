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
import argparse
import matplotlib.pyplot as plt
from utils import *

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--large_language_model_name', type=str,
                    default="meta-llama/Llama-3.2-1B-Instruct",
                    choices=["meta-llama/Llama-3.2-1B-Instruct",
                             "meta-llama/Llama-3.2-3B-Instruct",
                             "meta-llama/Llama-3.1-8B-Instruct",
                             "meta-llama/Llama-3.3-70B-Instruct"])
parser.add_argument('--sentence_transformer_name', type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2",
                    choices=["sentence-transformers/all-MiniLM-L6-v2",
                             "sentence-transformers/all-mpnet-base-v2",
                             "text-embedding-3-large",
                             "ada-embeddings"])
parser.add_argument('--debug', type=str2bool, default=True)
parser.add_argument('--device', type=str, default="cpu", choices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3',
                                                                  'cuda:4', 'cuda:5', 'cuda:6', 'cuda', 'cpu'])
parser.add_argument('--dataset_name', type=str, default="synthetic.json")
parser.add_argument('--tokenizer_padding_side', type=str, default='left')
parser.add_argument('--num_epoch', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_train_batch', type=int, default=20)
parser.add_argument('--num_valid_batch', type=int, default=20)
parser.add_argument('--stopper_patience', type=int, default=10)
parser.add_argument('--optimizer_learning_rate', type=float, default=0.001)
parser.add_argument('--optimizer_weight_decay', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--separate_query_head', type=str2bool, default=True)
parser.add_argument('--kb_scale_factor', type=float, default=None)
parser.add_argument('--kb_layer_frequency', type=int, default=3)
parser.add_argument('--save_model', type=str2bool, default=False)
parser.add_argument('--huggingface_accesstoken', type=str, default=None)
args = parser.parse_args()
check_args(args=args)

# main
torch.cuda.empty_cache()
torch.manual_seed(seed=args.seed)
np.random.seed(seed=args.seed)

dataloader = Dataloader(dataset_name=args.dataset_name, batch_size=args.batch_size, seed=args.seed)

#  C:\Users\86180\.cache\huggingface\

# SentenceEncoder
sentence_encoder = SentenceEncoder(model_name=args.sentence_transformer_name, device=args.device)

# AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.large_language_model_name,
                                          trust_remote_code=True,
                                          padding_side=args.tokenizer_padding_side,
                                          token=args.huggingface_accesstoken)
tokenizer.pad_token = tokenizer.eos_token

# LLM
if args.debug:
    llm_config = LlamaConfig(num_hidden_layers=2,
                             vocab_size=len(tokenizer.vocab),
                             pretraining_tp=1)
    llm = LlamaForCausalLM(config=llm_config)
else:
    llm = LlamaForCausalLM.from_pretrained(pretrained_model_name_or_path=args.large_language_model_name,
                                           token=args.huggingface_accesstoken)
llm.config.pretrained_model_name_or_path = args.large_language_model_name

# adapter
key_adapter = KeyAdapter(in_dim=sentence_encoder.out_dim, out_dim=llm.config.hidden_size)
value_adapter = ValueAdapter(in_dim=sentence_encoder.out_dim, out_dim=llm.config.hidden_size)
separate_query_linear = SeparateQueryLinear(in_dim=llm.config.hidden_size, out_dim=llm.config.hidden_size,
                                            num_layer=llm.config.num_hidden_layers)

# model
kblam = KBLaM(device=args.device,
              tokenizer=tokenizer,
              sentence_encoder=sentence_encoder,
              key_adapter=key_adapter,
              value_adapter=value_adapter,
              separate_query_head=args.separate_query_head,
              separate_query_linear=separate_query_linear,
              kb_layer_frequency=args.kb_layer_frequency,
              kb_scale_factor=args.kb_scale_factor,
              llm=llm)

# optimizer scheduler
trainable_parameters = list(kblam.key_adapter.parameters()) + list(kblam.value_adapter.parameters())
if kblam.separate_query_head:
    trainable_parameters += list(kblam.separate_query_linear.parameters())
optimizer = torch.optim.AdamW(params=trainable_parameters,
                              lr=args.optimizer_learning_rate,
                              weight_decay=args.optimizer_weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                       T_max=args.num_epoch,
                                                       eta_min=0.01 * args.optimizer_learning_rate)

# train utilities
stopper = EarlyStopping(patience=args.stopper_patience)
loss_recorder = LossRecoder()

num_train_batch = args.num_train_batch if args.num_train_batch else dataloader.num_train_batch
num_valid_batch = args.num_valid_batch if args.num_valid_batch else dataloader.num_valid_batch

# train cycle
start_time = datetime.now().strftime(('%Y-%m-%d %H-%M-%S'))
for epoch in range(0, args.num_epoch):
    if kblam.sentence_encoder.training:
        kblam.sentence_encoder.eval()
    if kblam.llm.training:
        kblam.llm.eval()
    if epoch == 0:
        check_gpu()

    # train
    optimizer.zero_grad()
    kblam.key_adapter.train()
    kblam.value_adapter.train()
    kblam.separate_query_linear.train()

    for batch_index in range(num_train_batch):
        batch_data, context_data = dataloader.train_dataloader(epoch=epoch, batch_index=batch_index)
        logits, true_label = kblam.forward(batch_data=batch_data, context_data=context_data)
        batch_loss = kblam.loss_function(logits=logits, true_label=true_label)
        batch_loss.backward()
        optimizer.step()
        loss_recorder.record(epoch=epoch, batch_index=batch_index, batch_train_loss=batch_loss.item())
    scheduler.step()

    # validation
    with torch.no_grad():
        kblam.eval()
        for batch_index in range(num_valid_batch):
            batch_data, context_data = dataloader.valid_dataloader(epoch=epoch, batch_index=batch_index)
            logits, true_label = kblam.forward(batch_data=batch_data, context_data=context_data)
            batch_loss = kblam.loss_function(logits=logits, true_label=true_label)
            loss_recorder.record(epoch=epoch, batch_index=batch_index, batch_valid_loss=batch_loss.item())

    train_loss, valid_loss = loss_recorder.get_epoch_loss(epoch=epoch)
    stopper.record(now_val_loss=valid_loss, model=kblam)

    if stopper.is_stop:
        break
    else:
        dataloader.shuffle_train_data()
        dataloader.shuffle_valid_data()

# test evalution
with torch.no_grad():
    kblam.load_state_dict(state_dict=stopper.best_model_parameter_state_dict)
    kblam.eval()
    kblam.llm.generation_config.pad_token_id = tokenizer.pad_token_id
    kblam.llm.generation_config.eos_token_id = tokenizer.eos_token_id

    pred_answer_NoKB_list = []
    pred_answer_UsKB_list = []
    true_answer_list = []

    for batch_index in range(dataloader.num_test_batch):
        batch_data, context_data = dataloader.test_dataloader(epoch=batch_index, batch_index=batch_index)
        pred_token_index_NoKB, pred_token_index_UsKB = kblam.forward(batch_data=batch_data,
                                                                     context_data=context_data,
                                                                     test=True)
        pred_token_index_NoKB = tokenizer.batch_decode(pred_token_index_NoKB, skip_special_tokens=False)
        pred_token_index_UsKB = tokenizer.batch_decode(pred_token_index_UsKB, skip_special_tokens=False)

        for index, data in enumerate(batch_data):
            # 1
            pred_answer_NoKB = pred_token_index_NoKB[index]
            pred_answer_NoKB = kblam.prune_text(sentence=pred_answer_NoKB)
            question = data["Q"]
            pred_answer_NoKB = pred_answer_NoKB.split(question)
            if len(pred_answer_NoKB) > 1:
                pred_answer_NoKB = pred_answer_NoKB[1]
            else:
                pred_answer_NoKB = ""
            pred_answer_NoKB_list.append(pred_answer_NoKB)

            # 2
            pred_answer_UsKB = pred_token_index_NoKB[index]
            pred_answer_UsKB = kblam.prune_text(sentence=pred_answer_UsKB)
            question = data["Q"]
            pred_answer_UsKB = pred_answer_UsKB.split(question)
            if len(pred_answer_UsKB) > 1:
                pred_answer_UsKB = pred_answer_UsKB[1]
            else:
                pred_answer_UsKB = ""
            pred_answer_UsKB_list.append(pred_answer_UsKB)

            # 3
            true_answer = data["A"]
            true_answer_list.append(true_answer)

        if args.debug: break

    rogue_score_NoKB, bert_score_NoKB = kblam.score_metrics(predictions=pred_answer_NoKB_list,
                                                            references=true_answer_list)
    rogue_score_UsKB, bert_score_UsKB = kblam.score_metrics(predictions=pred_answer_UsKB_list,
                                                            references=true_answer_list)
    print("rogue_score_NoKB", rogue_score_NoKB)
    print("bert_score_NoKB", bert_score_NoKB)
    print("rogue_score_UsKB", rogue_score_UsKB)
    print("bert_score_UsKB", bert_score_UsKB)

finish_time = datetime.now().strftime(('%Y-%m-%d %H-%M-%S'))
print(f"start:{start_time} finish:{finish_time}")

time = f"from_{start_time}_to_{finish_time}"

if stopper.is_stop and args.save_model:
    save_best_kblam(stopper, dataloader, kblam, time)

# save result
loss_recorder.draw()

# save result
kblam_config = get_model_hyperparameter(model=kblam)
llm_config = get_model_hyperparameter(model=kblam.llm)
sentence_encoder_config = get_model_hyperparameter(model=kblam.sentence_encoder)
key_adapter_config = get_model_hyperparameter(model=kblam.key_adapter)
value_adapter_config = get_model_hyperparameter(model=kblam.value_adapter)
separate_query_linear_config = get_model_hyperparameter(model=kblam)
global_config = get_model_hyperparameter(items=globals().items())

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
df.to_excel(f"{time}.xlsx")
