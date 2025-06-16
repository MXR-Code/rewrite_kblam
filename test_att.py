import os
import math
import numpy as np
from functools import partial

from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint

from transformers import AutoTokenizer
from transformers import PretrainedConfig

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING
from transformers.models.llama.modeling_llama import _CONFIG_FOR_DOC
from transformers.models.llama.modeling_llama import KwargsForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.models.llama.modeling_llama import repeat_kv

from transformers.utils import add_start_docstrings_to_model_forward
from transformers.utils import can_return_tuple
from transformers.utils import is_torch_flex_attn_available
from transformers.utils import replace_return_docstrings
from transformers.utils import is_torch_flex_attn_available

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.modeling_outputs import TokenClassifierOutput

from transformers.cache_utils import Cache
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import StaticCache

from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.deprecation import deprecate_kwarg
from llama_kblam import LlamaAttention_KBLaM



from transformers.models.llama.modeling_llama import LlamaConfig
config = LlamaConfig(num_hidden_layers=1, vocab_size=128009 + 1000)
layer_idx = 0
model = LlamaAttention_KBLaM(config, layer_idx)

# Generate dummy input data
batch_size = 10
query_seq_len = 50  # Example query length
hidden_dim = config.hidden_size
attention_mask = None  # Use None to skip the mask

# Create random input tensor for hidden_states
hidden_states = torch.randn(batch_size, query_seq_len, hidden_dim)

# Position embeddings (optional, set to None)
position_embeddings = (None, None)

# Create dummy position_ids
position_ids = torch.arange(query_seq_len).unsqueeze(0).repeat(batch_size, 1)

# Dummy knowledge base key/values (set kb_kvs to None if not used)
kb_kvs = (torch.randn(batch_size, 5, hidden_dim),  # Example knowledge base keys
          torch.randn(batch_size, 5, hidden_dim))  # Example knowledge base values

# Run the forward pass
attn_output, attn_weights, past_key_value = model(
    hidden_states=hidden_states,
    position_embeddings=position_embeddings,
    attention_mask=attention_mask,
    past_key_value=None,
    position_ids=position_ids,
    output_attentions=False,
    use_cache=False,
    kb_kvs=kb_kvs,
    kb_layer_frequency=3,
    dynamic_sparsify=False,
    topk_size=100,
    seperate_query_head=False,
    kb_scale_factor=None,
    save_attention_weights=False,
    attention_save_loc=None,
    attention_file_base_name=None)

# Display the shapes of the output tensors
print("Attention Output shape:", attn_output.shape)
if attn_weights is not None:
    print("Attention Weights shape:", attn_weights.shape)
else:
    print("Attention Weights shape: None")
