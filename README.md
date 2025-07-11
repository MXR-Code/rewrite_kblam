qpLLxNyWpIkgRpAsaHvHaGkTDwPHHXWupm

```bash
conda create -n mxr_kgllm python=3.10 -y
conda activate mxr_kgllm

# Cuda Version
nvcc --version
nvidia-smi

# GPU Pytorch
conda install pytorch pytorch-cuda=11.8 torchvision torchaudio torchdata -c pytorch -c nvidia

# Check Pytorch
python -c "import torch; print(torch.cuda.is_available())"

# Traceback (most recent call last):
# File "<stdin>", line 1, in <module>
# File "/data/machao/soft/anaconda3/envs/torch/lib/python3.8/site-packages/torch/__init__.py", line 197, in <module>
# from torch._C import * # noqa: F403
# ImportError: /data/machao/soft/anaconda3/envs/torch/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
pip install mkl==2024

# GPU Pytorch Geometric 
conda install pyg=*=*cu* pytorch-scatter pytorch-sparse pytorch-cluster pytorch-spline-conv -c pyg

# Huggingface Packages
conda install huggingface_hub transformers tokenizers datasets sentence-transformers -c conda-forge

# OpenAI API
conda install openai -c conda-forge
conda install azure-ai-agents azure-ai-contentsafety azure-ai-documentintelligence azure-ai-evaluation azure-ai-formrecognizer azure-ai-language-conversations azure-ai-language-questionanswering azure-ai-metricsadvisor azure-ai-ml azure-ai-textanalytics azure-ai-translation-document azure-ai-translation-text azure-ai-vision azure-appconfiguration azure-communication azure-confidentialledger azure-containerregistry azure-core azure-cosmos azure-data-tables azure-developer-loadtesting azure-digitaltwins-core azure-eventgrid azure-eventhub azure-health-deidentification azure-healthinsights azure-identity azure-iot-deviceupdate azure-keyvault azure-messaging-webpubsubclient azure-messaging-webpubsubservice azure-mgmt azure-monitor-ingestion azure-monitor-query azure-schemaregistry azure-search-documents azure-security-attestation azure-servicebus azure-storage -c microsoft

# Anthropic Claude API
conda install anthropic -c conda-forge

# Other
conda install wandb rich accelerate evaluate nltk rouge-score absl-py bert_score openpyxl selenium -c conda-forge

# Check Huggingface transformers
python -c "import transformers; print([d for d in dir(transformers) if "GenerationMixin" in d])"



conda activate base
conda remove --name mxr_kgllm --all -y
conda clean --all -y
```



```bash
conda activate mxr_kgllm
cd /home/dingqiyang/mxr/rewrite_kblam-main
nvidia-smi

python check_gpu.py

CUDA_VISIBLE_DEVICES=5

nohup \
python train_test.py \
--large_language_model_name "meta-llama/Llama-3.2-3B-Instruct" \
--sentence_transformer_name "sentence-transformers/all-MiniLM-L6-v2" \
--debug True \
--gpu_index "3,5" \
--dataset_name "synthetic.json" \
--tokenizer_padding_side "left" \
--num_epoch 10 \
--batch_size 10 \
--num_forward_batch 20 \
--stopper_patience 10 \
--optimizer_learning_rate 0.001 \
--optimizer_weight_decay 0.01 \
--seed 1 \
--separate_query_head True \
--kb_scale_factor False \
--kb_layer_frequency 3 \
--save_model False \
--huggingface_accesstoken hf-qpLLxNyWpIkgRpAsaHvHaGkTDwPHHXWupm
> output20250624.log 2>&1 &

cat output.log

```
