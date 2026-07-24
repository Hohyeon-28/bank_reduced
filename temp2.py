import time
from functools import partial

import torch
import timm
from models.early_vit_partial_routing import router_vit_small_patch16_224_depth12_6_e256_partial_mlp
from models.early_vit import router_vit_small_patch16_224_depth12_6_e256, vit_small_patch16_224_depth6, vit_small_patch16_224_depth12

# ✅ 동일한 실행 조건 강제
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.set_float32_matmul_precision("high")  # PyTorch 2.0 이상일 경우

def cuda_timestamp(sync=False, device=None):
    if sync:
        torch.cuda.synchronize(device=device)
    return time.perf_counter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_name = 'vit_small_patch16_224'
model_name = 'vit_small_patch16_224_depth12'
# model_name = "router_vit_small_patch16_224_depth12_6_e256_partial_mlp"
# model_name = 'router_vit_small_patch16_224_depth12_6_e256'
# model_name = 'vit_small_patch16_224_depth6'
model = timm.create_model(model_name, pretrained=False).to(device)
model.eval()
print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

batch = 1
x = torch.randn(batch, 3, 224, 224,device=device)
print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

time_fn = partial(cuda_timestamp, device=device)
t0 = time_fn(True)
with torch.no_grad():
    output = model(x)
t1 = time_fn(True)
print(f"{(t1-t0) * 10 ** 3:.3f} ms")

t0 = time_fn(True)
with torch.no_grad():
    output = model(x)
t1 = time_fn(True)
print(f"{(t1-t0) * 10 ** 3:.3f} ms")


print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
print("Max allocated:", torch.cuda.max_memory_allocated() / 1024**2, "MB")

# print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
# print("Max reserved:", torch.cuda.max_memory_reserved() / 1024**2, "MB")