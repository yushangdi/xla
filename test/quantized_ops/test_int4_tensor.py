import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
weight = torch.randint(-8, 7, (4, 2)).to(torch.int8)
x = torch.randn(3, 2).to(torch.bfloat16)
torch_out = F.linear(x, weight.to(x.dtype))

weight = weight.to(device)
x = x.to(device)

# torch_xla._XLAC._xla_mark_int4_tensor(weight, True)
# xla_out = F.linear(x, weight)
# hlo = torch_xla._XLAC._get_xla_tensors_hlo([xla_out])
# print(hlo)

# print(torch_out)
# print(xla_out)
