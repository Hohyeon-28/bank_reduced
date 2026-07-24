import torch
import timm
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

# ---- config ----
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_images = 128  # 빠르게 하기 위해 작게
imagenet_val_path = "../DATA/imageNet/val"  # 

# ---- dataloader ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])
val_dataset = datasets.ImageFolder(imagenet_val_path, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=max_images, shuffle=False, num_workers=4)

# ---- model ----
model = timm.create_model('vit_tiny_patch16_224', pretrained=True).to(device)
model.eval()

# ---- hook storage (1개만 저장) ----
msa_outputs = [None for _ in range(len(model.blocks))]

# ---- hook functions ----
def save_residual(module, input, output):
    for block in model.blocks:
        if block.norm1 is module:
            block.attn.input_residual = output

def save_msa_output(module, input, output):
    for i, block in enumerate(model.blocks):
        if block.attn is module:
            msa_out = block.attn.input_residual + block.drop_path1(output)
            # PATCH TOKEN만 저장
            patch_only = msa_out[:, 1:, :].reshape(-1, msa_out.shape[-1])  # [B×196, D]
            msa_outputs[i] = patch_only.detach()
            break

# ---- hook 등록 ----
for block in model.blocks:
    block.attn.input_residual = None
    block.norm1.register_forward_hook(save_residual)
    block.attn.register_forward_hook(save_msa_output)

# ---- inference ----
n_seen = 0
start = time.time()

with torch.no_grad():
    for images, _ in val_loader:
        images = images.to(device)
        _ = model(images)
        n_seen += images.size(0)
        if n_seen >= max_images:
            break

print(f"Collected MSA outputs for {n_seen} images in {time.time() - start:.2f}s")

# ---- CKA ----
def gram_linear(x): return x @ x.T
def center_gram(K):
    n = K.size(0)
    unit = torch.ones(n, n, device=K.device) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def cka(X, Y):
    X = X - X.mean(dim=0)
    Y = Y - Y.mean(dim=0)
    K = center_gram(gram_linear(X))
    L = center_gram(gram_linear(Y))
    hsic = (K * L).sum()
    norm_K = torch.norm(K)
    norm_L = torch.norm(L)
    if norm_K == 0 or norm_L == 0:
        return torch.tensor(0.0)
    return hsic / (norm_K * norm_L)

# ---- calculate CKA ----
n_layers = len(msa_outputs)
cka_matrix = torch.zeros(n_layers, n_layers)

for i in range(n_layers):
    for j in range(n_layers):
        cka_matrix[i, j] = cka(msa_outputs[i], msa_outputs[j])

# ---- plot ----
plt.figure(figsize=(10, 8))
sns.heatmap(
    cka_matrix.numpy(),
    cmap='rocket',       
    annot=True,
    fmt='.2f',
    linewidths=0.5,
    square=True,
    cbar=True
)
plt.title("CKA Similarity (MSA Output, Patch Only)")
plt.xlabel("Layer")
plt.ylabel("Layer")
plt.tight_layout()
plt.show()

