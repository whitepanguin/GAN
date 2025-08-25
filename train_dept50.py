# -----------------------------
# 🌻 [1] 공통 설정 및 환경 변수 정의
# -----------------------------
import os, math, json, random
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import utils
from dataclasses import dataclass, asdict
from copy import deepcopy
from typing import List, Literal, Optional

# 프로젝트 결과물이 저장될 기본 디렉토리 설정 (Colab에서도 사용 가능)
PROJECT_DIR = Path("./dept50")

# 학습 데이터가 들어있는 이미지 폴더 루트 (예: /content/sunflower_all/)
DATA_ROOT = Path("./bellflower/classA/")

# 결과 저장용 폴더 생성 (없으면 자동 생성)
RUNS_DIR = PROJECT_DIR / "runsdept5"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# 랜덤 시드 고정: 재현성 확보 (항상 같은 결과가 나올 수 있도록)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# 학습 장치 설정: GPU가 가능하면 CUDA, 아니면 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 만약 CUDA 사용 가능하다면 성능 향상을 위해 설정 (ConvNet에 적합)
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# -----------------------------
# 🌻 [2] Sunflower 이미지 Dataset 정의
# -----------------------------

# 사용할 이미지 확장자 목록 정의 (모든 일반 이미지 포맷 포함)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

# 이미지 경로를 재귀적으로 수집하는 함수
def collect_image_paths(root: Path) -> List[Path]:
    # root 하위 폴더까지 포함하여 모든 이미지 파일 경로를 리스트로 반환
    out = []
    for ext in IMG_EXTS:
        out += list(root.rglob(f"*{ext}"))
    return sorted(out)

# PyTorch Dataset 클래스 정의
class FlatImageDataset(torch.utils.data.Dataset):
    """
    모든 이미지를 고정된 해상도로 Resize + Crop + 정규화하여 반환하는 Dataset
    """
    def __init__(self, root_dir: Path, img_size: int):
        # 이미지 경로 리스트 수집
        self.paths = collect_image_paths(root_dir)
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root_dir}")

        # 이미지 전처리 Transform 정의: Resize → CenterCrop → Tensor → 정규화
        self.tfm = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # (C, H, W) 형태로 변환됨
            transforms.Normalize([0.5]*3, [0.5]*3)  # 픽셀값 [-1, 1]로 정규화
        ])

    def __len__(self):
        # 전체 이미지 개수 반환
        return len(self.paths)

    def __getitem__(self, i):
        # i번째 이미지 로딩 → RGB 변환 → 전처리 후 반환
        return self.tfm(Image.open(self.paths[i]).convert("RGB"))
# -----------------------------
# 🌻 [3] DataLoader 생성 함수
# -----------------------------

def make_loader(ds, bs, train=True):
    """
    주어진 Dataset을 배치 단위로 로딩할 DataLoader 생성
    - num_workers: 병렬 처리 워커 수 (Colab이면 2~4 권장)
    - pin_memory: CUDA 성능 향상 옵션
    - drop_last: 배치가 모자란 경우 버리기
    """
    return DataLoader(
        ds, bs,
        shuffle=train,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
# -----------------------------
# 🌻 [4-1] Generator 정의
# -----------------------------

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, ch=64):
        super().__init__()

        # G는 ConvTranspose2d 계층을 반복하여 4x4 → 8 → 16 → 32 → 64 → 128 로 업샘플링
        self.net = nn.Sequential(
            # 입력 latent vector(z)를 (ch*8)x4x4 형태로 투영
            nn.ConvTranspose2d(z_dim, ch * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True),  # (ch*8)x4x4

            nn.ConvTranspose2d(ch * 8, ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),  # (ch*4)x8x8

            nn.ConvTranspose2d(ch * 4, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),  # (ch*2)x16x16

            nn.ConvTranspose2d(ch * 2, ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),  # chx32x32

            nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(True),  # (ch//2)x64x64

            nn.ConvTranspose2d(ch // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # 출력 이미지: 3채널 RGB, 크기 128x128, 픽셀값 범위 [-1, 1]
        )

    def forward(self, z):
        # 입력 z는 (batch_size, z_dim) → reshape to (batch_size, z_dim, 1, 1)
        return self.net(z.view(z.size(0), z.size(1), 1, 1))
# -----------------------------
# 🌻 [4-2] Discriminator 정의
# -----------------------------

class Discriminator(nn.Module):
    def __init__(self, ch=64):
        super().__init__()

        # D는 Conv2d 계층을 반복하여 128x128 이미지를 1x1 로짓으로 줄임
        self.net = nn.Sequential(
            # 입력 이미지 (3, 128, 128) → (ch, 64, 64)
            nn.Conv2d(3, ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ch, ch * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),  # (ch*2)x32x32

            nn.Conv2d(ch * 2, ch * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),  # (ch*4)x16x16

            nn.Conv2d(ch * 4, ch * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 8),
            nn.LeakyReLU(0.2, inplace=True),  # (ch*8)x8x8

            nn.Conv2d(ch * 8, ch * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ch * 16),
            nn.LeakyReLU(0.2, inplace=True),  # (ch*16)x4x4

            nn.Conv2d(ch * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # 출력: (batch, 1, 1, 1) → flatten하면 로짓 1개로 변환됨
        )

    def forward(self, x):
        return self.net(x).view(-1)  # 로짓 하나로 flatten
# -----------------------------
# 🌻 [5] DCGAN-style 가중치 초기화
# -----------------------------

def weights_init(m):
    """
    DCGAN 논문에서 제안된 초기화 방식:
    - Conv, ConvTranspose: 평균 0, 표준편차 0.02 정규분포로 초기화
    - BatchNorm: 평균 1, 표준편차 0.02 (γ), 편향은 0 (β)
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
# -----------------------------
# 🌻 [6] 손실 함수 정의
# -----------------------------

# 생성자 손실 함수 (G)
def generator_loss(fake_logit):
    """
    G의 목표는 D를 속이는 것 (즉, D(fake)를 'real'로 분류하게 만들기)
    → BCE 기준, target은 1 (진짜처럼 보이도록)
    """
    target = torch.ones_like(fake_logit)  # 모든 값을 1로 설정
    return nn.BCEWithLogitsLoss()(fake_logit, target)

# 판별자 손실 함수 (D)
def discriminator_loss(real_logit, fake_logit):
    """
    D의 목표는 진짜(real)는 1, 가짜(fake)는 0으로 분류하는 것
    - real: target=1
    - fake: target=0
    """
    real_target = torch.ones_like(real_logit)
    fake_target = torch.zeros_like(fake_logit)
    real_loss = nn.BCEWithLogitsLoss()(real_logit, real_target)
    fake_loss = nn.BCEWithLogitsLoss()(fake_logit, fake_target)
    return real_loss + fake_loss

# -----------------------------
# 🌻 [7] 한 조합에 대해 전체 학습을 수행하는 함수
# -----------------------------

def train_one(cfg, dataset):
    # 🔸 실행 경로 준비
    RUN_DIR = RUNS_DIR / cfg.name
    SAMPLES_DIR = RUN_DIR / "samples"
    CKPT_DIR = RUN_DIR / "checkpoints"
    for d in [RUN_DIR, SAMPLES_DIR, CKPT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # 🔸 설정 저장 (나중에 분석용)
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 🔸 데이터로더 생성
    dl = make_loader(dataset, cfg.batch_size)

    # 🔸 모델 생성 및 초기화
    G = Generator(cfg.z_dim, cfg.g_ch).to(device)
    D = Discriminator(cfg.d_ch).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    # 🔸 Optimizer 설정 (Adam)
    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=cfg.betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=cfg.betas)

    # 🔸 고정 latent 벡터 (샘플용)
    fixed_z = torch.randn(cfg.n_samples, cfg.z_dim, device=device)

    # 🔸 학습 루프 시작
    for epoch in range(1, cfg.epochs + 1):
        G.train(); D.train()
        loss_G_epoch = 0.0
        loss_D_epoch = 0.0

        pbar = tqdm(dl, desc=f"[{cfg.name}] Epoch {epoch}/{cfg.epochs}", leave=False)

        for real in pbar:
            real = real.to(device)

            B = real.size(0)  # 현재 배치 크기

            # ------------------------
            # ① D 학습
            # ------------------------
            opt_D.zero_grad()

            # latent → fake 이미지 생성
            z = torch.randn(B, cfg.z_dim, device=device)
            fake = G(z)

            # D에 통과시켜 판별값(logit) 얻기
            real_logit = D(real)
            fake_logit = D(fake.detach())

            # 손실 계산 + 역전파
            loss_D = discriminator_loss(real_logit, fake_logit)
            loss_D.backward()
            opt_D.step()

            # ------------------------
            # ② G 학습
            # ------------------------
            opt_G.zero_grad()

            # 다시 G → D 통과
            gen = G(z)
            gen_logit = D(gen)

            # 손실 계산 + 역전파
            loss_G = generator_loss(gen_logit)
            loss_G.backward()
            opt_G.step()

            # 손실값 누적
            loss_G_epoch += loss_G.item() * B
            loss_D_epoch += loss_D.item() * B

        # 🔸 평균 손실 출력
        loss_G_epoch /= len(dataset)
        loss_D_epoch /= len(dataset)
        print(f"[{cfg.name}] Epoch {epoch:4d} | loss_G={loss_G_epoch:.4f} | loss_D={loss_D_epoch:.4f}")

        # 🔸 50 epoch마다: 1배치 생성 이미지 저장
        if epoch % cfg.sample_batch_every == 0:
            G.eval()
            with torch.no_grad():
                sample_dir = SAMPLES_DIR / f"ep_{epoch:04d}"
                sample_dir.mkdir(parents=True, exist_ok=True)

                z = torch.randn(cfg.batch_size, cfg.z_dim, device=device)
                fake = G(z).cpu()

                # 개별 이미지 저장
                for i in range(cfg.batch_size):
                    utils.save_image(
                        fake[i], sample_dir / f"fake_{i:04d}.png",
                        normalize=True, value_range=(-1, 1)
                    )

                # 그리드 저장 (nrow: 가로 행 수)
                utils.save_image(
                    fake, sample_dir / f"grid.png",
                    nrow=int(math.sqrt(cfg.batch_size)),
                    normalize=True, value_range=(-1, 1)
                )

        # 🔸 체크포인트 저장 (100 epoch마다)
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
# 🌻 [8] 5가지 실험 조합 정의
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
    lr_d: float = 2e-4
    betas: tuple = (0.5, 0.999)
    epochs: int = 500
    sample_batch_every: int = 50
    n_samples: int = 64

RUNS: List[RunConfig] = [
    RunConfig(name="exp1_default"),
    RunConfig(name="exp2_bigG", g_ch=96, z_dim=192),
    RunConfig(name="exp3_bigD", d_ch=128),
    RunConfig(name="exp4_lr_decay", lr_d=1e-4),
    RunConfig(name="exp5_deep", g_ch=128, d_ch=128, z_dim=256),
]
# -----------------------------
# 🌻 [9] 전체 실행
# -----------------------------
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


if __name__ == "__main__":
    # 데이터셋 준비
    dataset = FlatImageDataset(DATA_ROOT, img_size=128)

    # 모든 조합 순차 실행
    for cfg in RUNS:
        try:
            train_one(cfg, dataset)
        except Exception as e:
            print(f"[ERROR] Run failed: {cfg.name} => {repr(e)}")

