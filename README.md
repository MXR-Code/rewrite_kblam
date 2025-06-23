qpLLxNyWpIkgRpAsaHvHaGkTDwPHHXWupm

```bash
conda create -n mxr_kgllm python=3.10 -y
conda activate mxr_kgllm

# GPU Pytorch
nvcc --version
nvidia-smi
conda install cuda-toolkit=11.8 cuda-cupti=11.8 cuda-cudart=11.8 cuda-nvtx=11.8 cuda=11.8 cuda-compiler=11.8 cuda-runtime=11.8 cuda-nvprune=11.8 cuda-nvprune=11.8 cuda-libraries=11.8 -c nvidia
conda install pytorch pytorch-cuda=11.8 torchvision torchaudio torchdata -c pytorch

# CPU Pytorch
conda install pytorch torchvision torchaudio torchdata -c pytorch

# Check Pytorch
python -c "import torch; print(torch.cuda.is_available())"

# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# File "/data/machao/soft/anaconda3/envs/torch/lib/python3.8/site-packages/torch/__init__.py", line 197, in <module>
# from torch._C import * # noqa: F403
# ImportError: /data/machao/soft/anaconda3/envs/torch/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
pip install mkl==2024

# Huggingface Packages
conda install huggingface_hub transformers tokenizers datasets sentence-transformers azure-identity openai wandb rich accelerate evaluate nltk rouge-score absl-py bert_score openpyxl -c conda-forge

# Check Huggingface transformers
python -c "import transformers.GenerationMixin as aaa; print([d for d in dir(transformers) if "GenerationMixin" in d])"

# GPU Pytorch Geometric 
conda install pyg=*=*cu* pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg

# CPU Pytorch Geometric 
conda install pyg pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg

conda activate base
conda remove --name mxr_kgllm --all -y
conda clean --all -y
```

```bash
conda activate mxr_kgllm
cd /home/dingqiyang/mxr/rewrite_kblam-main

python check_gpu.py

CUDA_VISIBLE_DEVICES=5 nohup \

python train_test.py \
--debug False \
--device cuda:5 \
--num_train_batch 20 \
--num_valid_batch 20 \
--large_language_model_name meta-llama/Llama-3.2-1B-Instruct \
--sentence_transformer_name sentence-transformers/all-MiniLM-L6-v2 \
--dataset_name synthetic.json \
--tokenizer_padding_side left \
--num_epoch 10000 \
--batch_size 10 \
--stopper_patience 5 \
--optimizer_learning_rate 0.0001 \
--optimizer_weight_decay 0.0001 \
--seed 1 \--separate_query_head True \
--kb_layer_frequency 3 \
--save_model True \
--huggingface_accesstoken qpLLxNyWpIkgRpAsaHvHaGkTDwPHHXWupm

> output.log 2>&1 &

cat output.log

```
