import torch
from train import Config
from src.models import predrnn_v2, predrnn, attention_predrnn, rainpredrnn, rainpredrnn_v2
from thop import profile, clever_format


# Create a network and a corresponding input
configs = Config()
configs.device = 'cpu'
model = rainpredrnn_v2.RNN(len(configs.num_hidden), configs.num_hidden, configs).to(configs.device)
inp = torch.rand(2, 10, 128, 128, 1).to(configs.device)
mask = torch.rand(2, 4, 128, 128, 1).to(configs.device)

# Count the number of FLOPs
macs, params = profile(model, inputs=(inp, mask))
macs, params = clever_format([macs, params], "%.3f")
print(macs, params)
