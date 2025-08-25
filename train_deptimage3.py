# -*- coding: utf-8 -*-
"""
소규모(873장) 데이터셋에서 DCGAN 학습 안정화 + 기존 체크포인트에서 이어서 학습

적용한 변경점
- Data Augmentation 강화 (flip/rotation/color jitter)
- Label Smoothing (D의 real=0.9, fake=0.1)
- 학습률 비대칭: lr_g=2e-4, lr_d=1e-4
- 체크포인트 자동 재개: ./dept50/runsdept5/<run_name>/checkpoints/eXXXX.pt 중 가장 최신부터 이어서 학습

※ 기존에 학습하던 실험(exp1_default 등)의 아키텍처(G/D 채널 수)는 그대로 유지해야 재개가 가능합니다.
  아래 RUNS 리스트는 이전과 동일한 아키텍처로 설정되어 있습니다.
"""

import os, math, json, random, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from PIL import Image
from tqdm.auto import tqdm

# -----------------------------
# 🌻 [1] 공통 설정 및 환경 변수 정의
# -----------------------------
PROJECT_DIR = Path("./dept50")
DATA_ROOT = Path("./bellflower/classA/")
RUNS_DIR = PROJECT_DIR / "runsdept5"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# -----------------------------
# 🌻 [2] Dataset 정의 (강화된 Augmentation)
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

def collect_image_paths(root: Path) -> List[Path]:
    out = []
    for ext in IMG_EXTS:
        out += list(root.rglob(f"*{ext}"))
    return sorted(out)

class FlatImageDataset(torch.utils.data.Dataset):
    """
    모든 이미지를 고정 해상도로 변환. 소규모 데이터셋 안정화를 위해 Augmentation 강화.
    """
    def __init__(self, root_dir: Path, img_size: int):
        self.paths = collect_image_paths(root_dir)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        self.tfm = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tfm(Image.open(self.paths[i]).convert("RGB"))

# -----------------------------
# 🌻 [3] DataLoader
# -----------------------------

def make_loader(ds, bs, train=True):
    return DataLoader(
        ds, bs,
        shuffle=train,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )

# -----------------------------
# 🌻 [4-1] Generator
# -----------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=128, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ch * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ch // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))

# -----------------------------
# 🌻 [4-2] Discriminator (아키텍처는 기존과 동일 유지 — 재개 호환성)
# -----------------------------
class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 8, ch * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )
    def forward(self, x):
        return self.net(x).view(-1)

# -----------------------------
# 🌻 [5] DCGAN-style 가중치 초기화
# -----------------------------

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# -----------------------------
# 🌻 [6] 손실 함수 (Label Smoothing 적용)
# -----------------------------

def generator_loss(fake_logit):
    target = torch.ones_like(fake_logit)  # G는 D를 속이도록 1에 가깝게
    return nn.BCEWithLogitsLoss()(fake_logit, target)

# Label smoothing: real=0.9, fake=0.1
REAL_SMOOTH = 0.9
FAKE_SMOOTH = 0.1

def discriminator_loss(real_logit, fake_logit):
    real_target = torch.full_like(real_logit, REAL_SMOOTH)
    fake_target = torch.full_like(fake_logit, FAKE_SMOOTH)
    real_loss = nn.BCEWithLogitsLoss()(real_logit, real_target)
    fake_loss = nn.BCEWithLogitsLoss()(fake_logit, fake_target)
    return real_loss + fake_loss

# -----------------------------
# 🌻 [7] 체크포인트 유틸 (가장 최신 ckpt 불러오기)
# -----------------------------

def find_latest_ckpt(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("e*.pt"))
    if not cands:
        return None
    # 파일명 eXXXX.pt의 숫자로 정렬
    def epnum(p: Path):
        m = re.search(r"e(\d+)\.pt", p.name)
        return int(m.group(1)) if m else -1
    cands.sort(key=epnum)
    return cands[-1]

# -----------------------------
# 🌻 [8] 트레이닝 루프 (이어학습 지원)
# -----------------------------

def train_one(cfg, dataset):
    RUN_DIR = RUNS_DIR / cfg.name
    SAMPLES_DIR = RUN_DIR / "samples"
    CKPT_DIR = RUN_DIR / "checkpoints"
    for d in [RUN_DIR, SAMPLES_DIR, CKPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    dl = make_loader(dataset, cfg.batch_size)

    G = Generator(cfg.z_dim, cfg.g_ch).to(device)
    D = Discriminator(cfg.d_ch).to(device)

    start_epoch = 1
    total_epochs = cfg.epochs + getattr(cfg, 'extra_epochs', 0)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    if cfg.resume:
        latest = find_latest_ckpt(CKPT_DIR)
        if latest is not None:
            print(f"[INFO] Resuming from {latest}")
            ckpt = torch.load(latest, map_location=device)
            G.load_state_dict(ckpt["G"])  # 아키텍처 동일 전제
            D.load_state_dict(ckpt["D"])
            if "opt_G" in ckpt and "opt_D" in ckpt:
                try:
                    opt_G.load_state_dict(ckpt["opt_G"])
                    opt_D.load_state_dict(ckpt["opt_D"])
                except Exception as e:
                    print(f"[WARN] Optimizer state load failed: {e}. Recreating optimizers.")
                    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
                    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            print(f"[INFO] Resume start epoch: {start_epoch}")
        else:
            print("[INFO] No checkpoint found. Training from scratch.")
            G.apply(weights_init)
            D.apply(weights_init)
    else:
        G.apply(weights_init)
        D.apply(weights_init)

    fixed_z = torch.randn(cfg.n_samples, cfg.z_dim, device=device)

    for epoch in range(start_epoch, total_epochs + 1):
        G.train(); D.train()
        loss_G_epoch = 0.0
        loss_D_epoch = 0.0

        pbar = tqdm(dl, desc=f"[{cfg.name}] Epoch {epoch}/{total_epochs}", leave=False)

        for real in pbar:
            real = real.to(device)
            B = real.size(0)

            # ① D 학습
            opt_D.zero_grad()
            z = torch.randn(B, cfg.z_dim, device=device)
            fake = G(z)
            real_logit = D(real)
            fake_logit = D(fake.detach())
            loss_D = discriminator_loss(real_logit, fake_logit)
            loss_D.backward()
            opt_D.step()

            # ② G 학습
            opt_G.zero_grad()
            gen = G(z)  # 같은 z 다시 사용해도 무방
            gen_logit = D(gen)
            loss_G = generator_loss(gen_logit)
            loss_G.backward()
            opt_G.step()

            loss_G_epoch += loss_G.item() * B
            loss_D_epoch += loss_D.item() * B

        loss_G_epoch /= len(dataset)
        loss_D_epoch /= len(dataset)
        print(f"[{cfg.name}] Epoch {epoch:4d} | loss_G={loss_G_epoch:.4f} | loss_D={loss_D_epoch:.4f}")

        # 샘플 저장
        if epoch % cfg.sample_batch_every == 0:
            G.eval()
            with torch.no_grad():
                sample_dir = SAMPLES_DIR / f"ep_{epoch:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
                fake = G(z).cpu()
                for i in range(cfg.batch_size):
                    utils.save_image(
                        fake[i], sample_dir / f"fake_{i:04d}.png",
                        normalize=True, value_range=(-1, 1)
                    )
                utils.save_image(
                    fake, sample_dir / f"grid.png",
                    nrow=int(math.sqrt(cfg.batch_size)),
                    normalize=True, value_range=(-1, 1)
                )

        # 체크포인트 저장 (100 epoch마다)
        if epoch % 100 == 0:
            ckpt = {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "opt_G": opt_G.state_dict(),
                "opt_D": opt_D.state_dict(),
                "epoch": epoch,
                "cfg": asdict(cfg)
            }
            torch.save(ckpt, CKPT_DIR / f"e{epoch:04d}.pt")

# -----------------------------
# 🌻 [9] 실험 설정 (아키텍처는 기존과 동일하게 유지)
# -----------------------------
@dataclass
class RunConfig:
    name: str
    img_size: int = 128
    batch_size: int = 64
    z_dim: int = 128
    g_ch: int = 64
    d_ch: int = 64
    lr_g: float = 2e-4
    lr_d: float = 1e-4  # D를 더 낮게
    betas: tuple = (0.5, 0.999)
    epochs: int = 500   # 기본 목표 에폭(이전에 500으로 돌리셨음)
    extra_epochs: int = 0  # 이어학습 시 추가로 더 돌리고 싶으면 여기 늘리세요
    sample_batch_every: int = 50
    n_samples: int = 64
    resume: bool = True

RUNS: List[RunConfig] = [
    RunConfig(name="exp1_default"),
    # RunConfig(name="exp2_bigG", g_ch=96, z_dim=192),
    # RunConfig(name="exp3_bigD", d_ch=128),
    # RunConfig(name="exp4_lr_decay", lr_d=1e-4),
    RunConfig(name="exp5_deep", g_ch=128, d_ch=128, z_dim=256),
]

# -----------------------------
# 🌻 [10] 전체 실행
# -----------------------------
if __name__ == "__main__":
    dataset = FlatImageDataset(DATA_ROOT, img_size=128)
    for cfg in RUNS:
        try:
            train_one(cfg, dataset)
        except Exception as e:
            print(f"[ERROR] Run failed: {cfg.name} => {repr(e)}")
