from torch.nn import Linear, ModuleList
from torch.nn import Sequential
from torch.nn import LayerNorm
from torch.nn import Module
from transformers import FeatureExtractionMixin

class SeparateQueryLinear(Module, FeatureExtractionMixin):
    def __init__(self, in_dim, out_dim, num_layer):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layer = num_layer
        self.linear_list = ModuleList()
        for layer_index in range(num_layer):
            linear = Linear(self.in_dim, self.out_dim)
            self.linear_list.append(linear)

    def forward(self, hidden_embed=None, layer_index=0):
        assert layer_index < self.num_layer
        linear = self.linear_list[layer_index]
        query_embed = linear(hidden_embed)
        return query_embed