# %% [markdown]
# # **A Diffusion Model from Scratch in Pytorch**
# 
# In this notebook I want to build a very simple (as few code as possible) Diffusion Model for generating car images.

# %% [markdown]
# ### Investigating the dataset
# 
# As dataset we use the StandordCars Dataset, which consists of around 8000 images in the train set. Let's see if this is enough to get good results ;-)

# %%
import re
import torch
import kagglehub
import wandb
from torch.optim import Adam
import os, glob, shutil
import matplotlib
matplotlib.use('Agg')  # for remote server without display
from datetime import datetime

NUM_WORKERS = 4  # adjust based on your CPU requirements
PIN = torch.cuda.is_available()  # pin memory only if using GPU
torch.backends.cudnn.benchmark = True
USE_WANDB = True

TARGET_DIR = "/storage/work/szd399/DDPM_Images/data/stanford_cars"  # your custom folder
CACHE_DIR  = "/storage/work/szd399/DDPM_Images/data/.kagglehub_cache"  # keep cache in workdir too
OUTDIR = "/storage/work/szd399/DDPM_Images/outputs"
MODEL_DIR = "/storage/work/szd399/DDPM_Images/models"
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# %%
def has_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png")
    for ext in exts:
        if glob.glob(os.path.join(root, "**", ext), recursive=True):
            return True
    return False

def ensure_stanford_cars_dataset(target_dir=TARGET_DIR):
    os.makedirs(target_dir, exist_ok=True)
    marker = os.path.join(target_dir, ".READY")

    if os.path.exists(marker):
        print(f"Dataset already present at: {target_dir}")
        return target_dir

    # 1) If already present locally, use it
    if has_images(target_dir):
        open(marker, "w").close()
        print(f"Dataset already present at: {target_dir}")
        return target_dir

    # 2) Otherwise download via kagglehub (to cache)
    os.environ["KAGGLEHUB_CACHE_DIR"] = CACHE_DIR
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Dataset not found locally. Downloading via kagglehub...")
    downloaded_path = kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset")
    print("Downloaded to:", downloaded_path)

    # 3) Copy dataset into your target_dir
    #    If kagglehub returns a directory that already contains images, copy it.
    if not has_images(downloaded_path):
        raise RuntimeError(f"Downloaded path doesn't seem to contain images: {downloaded_path}")

    print(f"Copying dataset into: {target_dir}")
    shutil.copytree(downloaded_path, target_dir, dirs_exist_ok=True)

    # 4) Final sanity check
    if not has_images(target_dir):
        raise RuntimeError(f"Copy finished but no images found in: {target_dir}")
    
    open(marker, "w").close()
    print(f"Dataset ready at: {target_dir}")
    return target_dir


# Use this as your dataset root
path = ensure_stanford_cars_dataset()
print("Using dataset path:", path)

# %%
print("Base path:", path)
for root, dirs, files in os.walk(path):
    print(root)
    # just show top few entries then break to avoid huge spam
    dirs[:] = dirs[:3]
    break


# %%
import glob
from PIL import Image
from torch.utils.data import Dataset

class StanfordCarsKaggle(Dataset):
  def __init__(self, root, transform=None):
    self.root = root
    self.transform = transform
    exts = ("*.jpg", "*.jpeg", "*.png")
    self.image_paths = []
    for ext in exts:
      self.image_paths.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))
    self.image_paths.sort()

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    img = Image.open(img_path).convert("RGB")
    if self.transform is not None:
      img = self.transform(img)
    label = 0
    return img, label

# %%
import matplotlib.pyplot as plt

def show_images(dataset, num_samples=20, cols=4):
  """ Plot some samples from the dataset """
  plt.figure(figsize=(15,15))
  for i, img in enumerate(dataset):
    if i == num_samples:
      break
    plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
    plt.imshow(img[0])
    # plt.axis('off')
  # plt.show()

data = StanfordCarsKaggle(root=path)
# show_images(data)

# %% [markdown]
# ### Building the Diffusion Model

# %% [markdown]
# ### Step 1: The forward process = Noise scheduler

# %%
import torch.nn.functional as F

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
  return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
  batch_size = t.shape[0]
  vals = vals.to(t.device)
  out = vals.gather(-1, t)
  return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
  noise = torch.randn_like(x_0)
  sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
  # mean + variance
  return sqrt_alphas_cumprod_t * x_0 \
  + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

# %% [markdown]
# Let's test it on our dataset

# %%
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

IMG_SIZE = 64
BATCH_SIZE = 128

def load_transformed_dataset():
  data_transforms = [
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),  # Scales data into [0, 1]
      transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
  ]
  data_transform = transforms.Compose(data_transforms)

  # Below version is depreciated
  # train = torchvision.datasets.StanfordCars(root=".", download=True,
  #                                           transform=data_transform)
  # test = torchvision.datasets.StanfordCars(root=".", download=True,
  #                                          transform=data_transform, split='test')
  # return torch.utils.data.ConcatDataset([train, test])

  # Use this new version rather
  dataset = StanfordCarsKaggle(root=path, transform=data_transform)
  return dataset

def show_tensor_image(image):
  reverse_trasforms = transforms.Compose([
      transforms.Lambda(lambda t: (t + 1) / 2),
      transforms.Lambda(lambda t: t.permute(1, 2, 0)),   # CHW to HWC
      transforms.Lambda(lambda t: t * 255.0),
      transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
      transforms.ToPILImage(),
  ])

  # Take the first image of batch
  if len(image.shape) == 4:
    image = image[0, :, :, :]
  plt.imshow(reverse_trasforms(image))

data = load_transformed_dataset()
dataloader = DataLoader(data, 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, 
                        drop_last=True,
                        num_workers=NUM_WORKERS, 
                        pin_memory=PIN, 
                        persistent_workers=(NUM_WORKERS > 0))

# %% [markdown]
# Simulate the forward diffusion

# %%
image = next(iter(dataloader))[0]

plt.figure(figsize=(15, 2))
plt.axis('off')
num_images = 10
stepsize = int(T/ num_images)

for idx in range(0, T, stepsize):
  t = torch.tensor([idx], dtype=torch.long, device=image.device)
  plt.subplot(1, num_images, int(idx / stepsize) + 1)
  img, noise = forward_diffusion_sample(image, t)
  show_tensor_image(img)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/forward_diffusion_demo.png", dpi=200)
plt.close()

# %% [markdown]
# ## Step 2: The Backward process = U-Net

# %%
from torch import nn
import math

class Block(nn.Module):
  def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
    super().__init__()
    self.time_mlp = nn.Linear(time_emb_dim, out_ch)
    if up:
      self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
      self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
    else:
      self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
      self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.bnorm1 = nn.BatchNorm2d(out_ch)
    self.bnorm2 = nn.BatchNorm2d(out_ch)
    self.relu = nn.ReLU()

  def forward(self, x, t):
    # First Conv
    h = self.bnorm1(self.relu(self.conv1(x)))
    # Time embedding
    time_emb = self.relu(self.time_mlp(t))
    # Extend last 2 dimensions
    time_emb = time_emb[(...,) + (None, ) * 2]
    # Add time channel
    h = h + time_emb
    # Second Conv
    h = self.bnorm2(self.relu(self.conv2(h)))
    # Down or Upsample
    return self.transform(h)

class SinusiodalPositionEmbeddings(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, time):
    device = time.device
    half_dim = self.dim // 2
    embeddings = math.log(10000) / (half_dim - 1)
    embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
    embeddings = time[:, None] * embeddings[None, :]
    embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
    return embeddings

class SimpleUnet(nn.Module):
  def __init__(self):
    super().__init__()
    image_channels = 3
    down_channels = (64, 128, 256, 512, 1024)
    up_channels = (1024, 512, 256, 128, 64)
    out_dim = 3
    time_emb_dim = 32

    # Time embedding
    self.time_mlp = nn.Sequential(
        SinusiodalPositionEmbeddings(time_emb_dim),
        nn.Linear(time_emb_dim, time_emb_dim),
        nn.ReLU()
    )

    # Initial projection
    self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

    # Downsample
    self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i + 1], \
                                      time_emb_dim) \
                                for i in range(len(down_channels) - 1)])
    # Upsample
    self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                    time_emb_dim, up=True) \
                              for i in range(len(up_channels) - 1)])
    self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

  def forward(self, x, timestep):
    #Embedded time
    t = self.time_mlp(timestep)
    # Initial conv
    x = self.conv0(x)
    # Unet
    residual_inputs = []
    for down in self.downs:
      x = down(x, t)
      residual_inputs.append(x)
    for up in self.ups:
      residual_x = residual_inputs.pop()
      # Add residual x as additional channels
      x = torch.cat((x, residual_x), dim=1)
      x = up(x, t)
    return self.output(x)

model = SimpleUnet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model

# %% [markdown]
# ## Step 3: The loss

# %%
def get_loss(model, x_0, t):
  x_noisy, noise = forward_diffusion_sample(x_0, t)
  noise_pred = model(x_noisy, t)
  return F.l1_loss(noise, noise_pred)

# %% [markdown]
# ## Sampling

# %%
@torch.no_grad()
def sample_timestep(x, t):
  betas_t = get_index_from_list(betas, t, x.shape)
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, x.shape
  )
  sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

  # Call model (current image - noise prediction)
  model_mean = sqrt_recip_alphas_t * (
      x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
  )
  posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

  if t.item() == 0:
    # The t's are offset from the t's in the paper
    out = model_mean
  else:
    noise = torch.randn_like(x)
    out = model_mean + torch.sqrt(posterior_variance_t) * noise

  return out

@torch.no_grad()
def sample_plot_image(save_path):
  # important: disable BN updates, dropout (if any)
  was_training  = model.training
  model.eval()

  # Sample noise
  img_size = IMG_SIZE
  img = torch.randn((1, 3, img_size, img_size), device=device)
  plt.figure(figsize=(15, 2))
  plt.axis('off')
  num_images = 10
  stepsize = int(T / num_images)

  plot_index = 0
  for i in range(0, T)[::-1]:
    t = torch.full((1,), i, device=device, dtype=torch.long)
    img = sample_timestep(img, t)
    img = torch.clamp(img, -1.0, 1.0)
    if i % stepsize == 0:
      plot_index += 1
      plt.subplot(1, num_images, plot_index)
      show_tensor_image(img.detach().cpu())
  # plt.show()
  plt.tight_layout()
  plt.savefig(save_path, dpi=200)
  plt.close()

  if was_training:
    model.train()

# %% [markdown]
# ## Training

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# move diffusion schedule tensors to device
betas = betas.to(device)
alphas = alphas.to(device)
alphas_cumprod = alphas_cumprod.to(device)
alphas_cumprod_prev = alphas_cumprod_prev.to(device)
sqrt_recip_alphas = sqrt_recip_alphas.to(device)
sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device)
sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device)
posterior_variance = posterior_variance.to(device)

model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)

epochs = 500 # Try more
start_epoch = 0

# ---- 1) Resume FIRST: find latest ckpt + load run_id ----
ckpts = glob.glob(os.path.join(MODEL_DIR, "ckpt_epoch_*.pt"))

run_id = None
if len(ckpts) > 0:
  def epoch_num(p):
    m = re.search(r"ckpt_epoch_(\d+)\.pt$", os.path.basename(p))
    return int(m.group(1)) if m else -1

  latest = max(ckpts, key=epoch_num)
  ckpt = torch.load(latest, map_location=device)

  model.load_state_dict(ckpt["model"])
  optimizer.load_state_dict(ckpt["optim"])
  start_epoch = ckpt["epoch"] + 1
  run_id = ckpt.get("wandb_run_id", None)
  print(f"Resuming from {latest} (start_epoch={start_epoch})")
else:
    print("No checkpoint found. Starting from scratch.")

# ---- 2) THEN wandb.init, using that run_id ----
if USE_WANDB:
  wandb.init(
    project="ddpm_stanfordcars",
    # id=run_id,          # None on first run; existing id when resuming
    id=None,
    resume="allow",
    name=f"unet64_T{T}_bs{BATCH_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    config={
      "img_size": IMG_SIZE,
      "batch_size": BATCH_SIZE,
      "T": T,
      "lr": 0.001,
      "epochs": epochs,
      "model": "SimpleUnet",
      "loss": "L1(noise, noise_pred)",
    }
  )

# ---- 3) Training loop ----
for epoch in range(start_epoch, epochs):
  model.train()
  print(f"Epoch {epoch}")
  for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    x0 = batch[0].to(device, non_blocking=True)
    t = torch.randint(0, T, (x0.size(0),), device=device).long()
    loss = get_loss(model, x0, t)
    loss.backward()
    optimizer.step()

    global_step = epoch * len(dataloader) + step

    if USE_WANDB and step % 10 == 0:
      wandb.log({
        "train/loss_l1": loss.item(),
        "train/lr": optimizer.param_groups[0]["lr"],
        "epoch": epoch,
      }, step=global_step)

    if epoch % 50 == 0 and step == 0:
      print(f"At Epoch {epoch} | step {step:03d} Loss: {loss.item()}")
      sample_plot_image(f"{OUTDIR}/sample_epoch_{epoch}.png")
      if USE_WANDB:
        wandb.log({"samples": wandb.Image(f"{OUTDIR}/sample_epoch_{epoch}.png")}, 
                  step=global_step)
      ckpt_path = f"{MODEL_DIR}/ckpt_epoch_{epoch}.pt"
      torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "wandb_run_id": wandb.run.id if USE_WANDB else None,
      }, ckpt_path)


if USE_WANDB:
  wandb.finish()
# %%



