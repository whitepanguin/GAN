import os, math, glob, random, argparse, time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms
import torchvision.utils as vutils
from tqdm import tqdm
from torch.nn.utils import spectral_norm as SN

# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainCfg:
    data_root: str = "./bellflower"
    outdir: str = "./outputs_dcgan64"
    epochs: int = 2000
    batch_size: int = 128
    workers: int = 4
    image_size: int = 64
    nc: int = 3
    nz: int = 128
    ngf: int = 64
    ndf: int = 64

    # Optim (TTUR)
    lrG: float = 2e-4
    lrD: float = 1e-4
    beta1: float = 0.0
    beta2: float = 0.99

    # Stabilization
    inst_noise_sigma0: float = 0.06
    r1_every: int = 16
    r1_gamma: float = 5.0
    ema_decay: float = 0.999
    use_spectral_norm: bool = True
    use_mbstd: bool = True  # 간단 minibatch-stddev 채널 추가 (옵션)
    aug_flip: bool = True

    # Save / Resume
    save_intermediate_samples: bool = True
    save_intermediate_ckpt: bool = False
    sample_every: int = 200
    ckpt_every: int = 10
    resume: str = ""  # best.pt 경로 지정하면 이어서 학습

    # Misc
    seed: int = 42
    deterministic: bool = False  # True면 속도↓, 재현성↑

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

def weights_init_dcgan(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.zeros_(m.bias.data)

def apply_spectral_norm(module):
    """Conv/Linear에 spectral norm 적용 (중복 적용 방지)."""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            already_sn = any(name.endswith("weight_u") for name, _ in m.named_buffers())
            if not already_sn:
                try:
                    SN(m)
                except Exception:
                    pass

# 인스턴스 노이즈: 더 빨리 줄어들도록
def add_instance_noise(x, epoch, max_epoch, sigma0=0.06):
    p = min(1.0, epoch / (0.1 * max_epoch + 1e-6))
    sigma = sigma0 * (1.0 - p)
    if sigma <= 0: return x
    return x + sigma * torch.randn_like(x)


def diff_augment(x, image_size):
    # 아주 가벼운 geometric + brightness
    if torch.rand(1, device=x.device) < 0.8:
        B, C, H, W = x.shape
        dx = torch.randint(-2, 3, (B,), device=x.device).float() / max(W,1)
        dy = torch.randint(-2, 3, (B,), device=x.device).float() / max(H,1)
        theta = torch.zeros(B, 2, 3, device=x.device)
        theta[:,0,0] = 1; theta[:,1,1] = 1
        theta[:,0,2] = dx; theta[:,1,2] = dy
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="reflection")
    if torch.rand(1, device=x.device) < 0.8:
        scale = (0.9 + 0.2*torch.rand(x.size(0),1,1,1, device=x.device))
        x = (x * scale).clamp_(-1, 1)
    return x

class MinibatchStdDev(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        if B < 2:
            return x
        std = x.view(B, C, -1).std(dim=2, unbiased=False).mean()
        stdmap = std.view(1,1,1,1).expand(B,1,H,W)
        return torch.cat([x, stdmap], 1)

# ----------------------------
# Models (DCGAN)
# ----------------------------
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # [-1,1]
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc, use_mbstd=False):
        super().__init__()
        blocks = [
            # no BN in first block (DCGAN rule)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.features = nn.Sequential(*blocks)
        self.mbstd = MinibatchStdDev() if use_mbstd else nn.Identity()
        in_ch = ndf*8 + (1 if use_mbstd else 0)
        self.tail = nn.Sequential(
            nn.Conv2d(in_ch, 1, 4, 1, 0, bias=False)  # logits
        )

    def forward(self, x):
        h = self.features(x)
        h = self.mbstd(h)
        out = self.tail(h).view(-1, 1)
        return out  # logits (B,1)

# ----------------------------
# Checkpoint helpers
# ----------------------------
def save_ckpt(path, netG, netD, optG, optD, ema_G, epoch, global_step, best_lossG, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optG": optG.state_dict(),
        "optD": optD.state_dict(),
        "ema_G": (ema_G.state_dict() if ema_G is not None else None),
        "epoch": epoch,
        "global_step": global_step,
        "lossG": best_lossG,
        "cfg": cfg.__dict__,
    }, path)

def load_ckpt(path, device, netG, netD, optG, optD):
    ckpt = torch.load(path, map_location=device)
    if "netG" in ckpt: netG.load_state_dict(ckpt["netG"], strict=False)
    if "netD" in ckpt: netD.load_state_dict(ckpt["netD"], strict=False)
    if "optG" in ckpt:
        try: optG.load_state_dict(ckpt["optG"])
        except: pass
    if "optD" in ckpt:
        try: optD.load_state_dict(ckpt["optD"])
        except: pass
    ema_state = ckpt.get("ema_G", None)
    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("global_step", 0)
    best_lossG = ckpt.get("lossG", float("inf"))
    return ema_state, epoch, step, best_lossG

# ----------------------------
# Training
# ----------------------------
def main(cfg: TrainCfg):
    set_seed(cfg.seed, cfg.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Data
    tfms = []
    tfms.append(transforms.RandomResizedCrop(
        cfg.image_size, scale=(0.8, 1.0),
        interpolation=transforms.InterpolationMode.BILINEAR
    ))
    if cfg.aug_flip:
        tfms.append(transforms.RandomHorizontalFlip(p=0.5))
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize([0.5]*cfg.nc, [0.5]*cfg.nc),
    ]
    dataset = datasets.ImageFolder(root=cfg.data_root, transform=transforms.Compose(tfms))
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.workers, pin_memory=True, drop_last=True)
    print(f"[Data] {len(dataset)} images, {len(loader)} iters/epoch")

    # Models
    netG = Generator(cfg.nz, cfg.ngf, cfg.nc).to(device)
    # Discriminator 생성 시
    netD = Discriminator(cfg.ndf, cfg.nc, use_mbstd=cfg.use_mbstd).to(device)
    netG.apply(weights_init_dcgan)
    netD.apply(weights_init_dcgan)
    if cfg.use_spectral_norm:
        apply_spectral_norm(netD)

    # EMA
    ema_G = Generator(cfg.nz, cfg.ngf, cfg.nc).to(device)
    ema_G.load_state_dict(netG.state_dict())
    for p in ema_G.parameters():
        p.requires_grad_(False)

    # 기존 update_ema 함수를 아래로 교체
    def update_ema(step, total_steps, base_decay=cfg.ema_decay):
        # 초반엔 decay 낮춰 ‘워밍업’ → EMA가 초기값에 붙어죽는 것 방지
        warmup_frac = 0.02  # 앞 2% 스텝은 낮은 decay
        warmup_decay = 0.9
        decay = warmup_decay if step < total_steps * warmup_frac else base_decay
        with torch.no_grad():
            for p, p_ema in zip(netG.parameters(), ema_G.parameters()):
                p_ema.mul_(decay).add_(p, alpha=1-decay)
            # BN running stats(버퍼)는 그냥 복사
            for b, b_ema in zip(netG.buffers(), ema_G.buffers()):
                b_ema.copy_(b)


    # Optims & Loss
    optD = torch.optim.Adam(netD.parameters(), lr=cfg.lrD, betas=(cfg.beta1, cfg.beta2))
    optG = torch.optim.Adam(netG.parameters(), lr=cfg.lrG, betas=(cfg.beta1, cfg.beta2))
    criterion = nn.BCEWithLogitsLoss()
    scalerD = GradScaler(enabled=(device.type == "cuda"))
    scalerG = GradScaler(enabled=(device.type == "cuda"))

    # Out dirs
    os.makedirs(cfg.outdir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.outdir, "ckpt")
    samples_dir = os.path.join(cfg.outdir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Resume
    start_epoch, global_step = 0, 0
    best_lossG = float("inf")
    if cfg.resume and os.path.isfile(cfg.resume):
        print(f"[Resume] loading {cfg.resume}")
        ema_state, start_epoch, global_step, best_lossG = load_ckpt(
            cfg.resume, device, netG, netD, optG, optD
        )
        if ema_state is not None:
            try:
                ema_G.load_state_dict(ema_state)
                print("[Resume] EMA loaded.")
            except:
                print("[Resume] EMA load failed (ignored).")
        print(f"[Resume] epoch={start_epoch}, step={global_step}, best_lossG={best_lossG:.4f}")

    # Fixed noise for snapshot (옵션)
    fixed_z = torch.randn(64, cfg.nz, 1, 1, device=device)

    # Train
    for epoch in range(start_epoch, cfg.epochs):
        netG.train(); netD.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for real, _ in pbar:
            real = real.to(device, non_blocking=True)
            b = real.size(0)

            # ------------------ D ------------------
            optD.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                # real
                real_in = add_instance_noise(real, epoch, cfg.epochs, sigma0=cfg.inst_noise_sigma0)
                real_in = diff_augment(real_in, cfg.image_size)
                out_real = netD(real_in)
                # D real 타겟
                real_t = torch.ones_like(out_real) * (0.95 + 0.02 * torch.rand_like(out_real))
                loss_real = criterion(out_real, real_t)

                # fake
                z = torch.randn(b, cfg.nz, 1, 1, device=device)
                with torch.no_grad():
                    fake = netG(z)
                fake_in = add_instance_noise(fake, epoch, cfg.epochs, sigma0=cfg.inst_noise_sigma0)
                fake_in = diff_augment(fake_in, cfg.image_size)
                out_fake = netD(fake_in)
                # D fake 타겟
                fake_t = (0.02 * torch.rand_like(out_fake)).clamp_(0, 0.05)
                loss_fake = criterion(out_fake, fake_t)

                lossD = loss_real + loss_fake

            scalerD.scale(lossD).backward()

            # Lazy R1 on real
            if cfg.r1_every > 0 and (global_step % cfg.r1_every == 0):
                with autocast(enabled=False):
                    real_in_req = real_in.detach().requires_grad_(True)
                    out_real_r1 = netD(real_in_req)
                    grad = torch.autograd.grad(outputs=out_real_r1.sum(),
                                               inputs=real_in_req,
                                               create_graph=True, retain_graph=True, only_inputs=True)[0]
                    r1 = grad.pow(2).view(grad.size(0), -1).sum(1).mean()
                    r1_term = cfg.r1_gamma * 0.5 * r1
                scalerD.scale(r1_term).backward()

            scalerD.step(optD)
            scalerD.update()

            # ------------------ G ------------------
            optG.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                z = torch.randn(b, cfg.nz, 1, 1, device=device)
                gen = netG(z)
                gen_in = add_instance_noise(gen, epoch, cfg.epochs, sigma0=cfg.inst_noise_sigma0)
                gen_in = diff_augment(gen_in, cfg.image_size)
                out = netD(gen_in)
                lossG = criterion(out, torch.ones_like(out))  # non-saturating

            scalerG.scale(lossG).backward()
            scalerG.step(optG)
            scalerG.update()

            # 기존: update_ema(cfg.ema_decay)
            update_ema(global_step, total_steps=cfg.epochs * len(loader))


            global_step += 1
            pbar.set_postfix(lossD=float(lossD.detach().cpu()),
                             lossG=float(lossG.detach().cpu()))

            # Optional: save intermed.
            if cfg.save_intermediate_ckpt and (global_step % (cfg.ckpt_every * len(loader)) == 0):
                save_ckpt(os.path.join(ckpt_dir, f"ckpt_e{epoch:04d}.pt"),
                          netG, netD, optG, optD, ema_G,
                          epoch, global_step, best_lossG, cfg)

        # Best by lossG (epoch end 기준)
        if lossG.item() < best_lossG:
            best_lossG = lossG.item()
            save_ckpt(os.path.join(ckpt_dir, "best.pt"),
                      netG, netD, optG, optD, ema_G,
                      epoch, global_step, best_lossG, cfg)
            print(f"[best.pt] updated (lossG={best_lossG:.4f})")

        # epoch 끝에서, 50ep마다
        if cfg.save_intermediate_samples and ((epoch+1) % cfg.sample_every == 0 or epoch < 3):
            with torch.no_grad():
                netG.eval(); ema_G.eval()
                fake_g   = netG(fixed_z).cpu()
                fake_ema = ema_G(fixed_z).cpu()
            # 정규화 혼동 방지: [-1,1] -> [0,1] 변환 후 저장
            def to01(x): return x.add(1).div(2).clamp_(0,1)
            vutils.save_image(to01(fake_g),   os.path.join(samples_dir, f"e{epoch:04d}_G.png"),   nrow=8)
            vutils.save_image(to01(fake_ema), os.path.join(samples_dir, f"e{epoch:04d}_EMA.png"), nrow=8)

    # Final save
    save_ckpt(os.path.join(ckpt_dir, "final.pt"),
              netG, netD, optG, optD, ema_G,
              cfg.epochs-1, global_step, best_lossG, cfg)
    print("[Done] best.pt + final.pt saved at", ckpt_dir)

    # Optional: sample 64 from EMA after training
    with torch.no_grad():
        ema_G.eval()
        z = torch.randn(64, cfg.nz, 1, 1, device=device)
        imgs = ema_G(z).cpu()
    vutils.save_image(imgs, os.path.join(cfg.outdir, "final_ema_grid.png"),
                      nrow=8, normalize=True, value_range=(-1,1))
    print("[Sample] final_ema_grid.png saved")

# ----------------------------
# Sampling from best.pt
# ----------------------------
def load_generator_from_best(best_path, device, nz, ngf, nc):
    ckpt = torch.load(best_path, map_location=device)
    G = Generator(nz, ngf, nc).to(device)
    if ckpt.get("ema_G", None) is not None:
        try:
            G.load_state_dict(ckpt["ema_G"], strict=False)
            print("[Sample] Loaded EMA generator.")
            return G.eval()
        except:
            pass
    G.load_state_dict(ckpt["netG"], strict=False)
    print("[Sample] Loaded netG generator.")
    return G.eval()

def generate_from_best(best_path, outdir, n_images=1000, batch=64, nz=128, ngf=64, nc=3, prefix="gen"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = load_generator_from_best(best_path, device, nz, ngf, nc)
    os.makedirs(outdir, exist_ok=True)
    steps = math.ceil(n_images / batch)
    idx = 0
    with torch.no_grad():
        for _ in tqdm(range(steps), desc="Generate (best.pt)"):
            cur = min(batch, n_images - idx)
            z = torch.randn(cur, nz, 1, 1, device=device)
            imgs = G(z).cpu()
            for i in range(cur):
                p = os.path.join(outdir, f"{prefix}_{idx+i:06d}.png")
                vutils.save_image(imgs[i], p, normalize=True, value_range=(-1,1))
            idx += cur
    print("[Generate] done:", outdir)

# ----------------------------
# CLI
# ----------------------------
def str2bool(v: Optional[str]) -> bool:
    if v is None: return False
    return v.lower() in ("1","true","t","yes","y")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--outdir", type=str, default="./outputs_dcgan64")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--lrG", type=float, default=2e-4)
    parser.add_argument("--lrD", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--inst_noise_sigma0", type=float, default=0.1)
    parser.add_argument("--r1_every", type=int, default=16)
    parser.add_argument("--r1_gamma", type=float, default=10.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--use_spectral_norm", type=str, default="true")
    parser.add_argument("--use_mbstd", type=str, default="false")
    parser.add_argument("--aug_flip", type=str, default="true")
    parser.add_argument("--save_intermediate_samples", type=str, default="false")
    parser.add_argument("--save_intermediate_ckpt", type=str, default="false")
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--ckpt_every", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", type=str, default="false")
    args = parser.parse_args()

    cfg = TrainCfg(
        data_root=args.data_root,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        nc=args.nc,
        nz=args.nz,
        ngf=args.ngf,
        ndf=args.ndf,
        lrG=args.lrG,
        lrD=args.lrD,
        beta1=args.beta1,
        beta2=args.beta2,
        inst_noise_sigma0=args.inst_noise_sigma0,
        r1_every=args.r1_every,
        r1_gamma=args.r1_gamma,
        ema_decay=args.ema_decay,
        use_spectral_norm=str2bool(args.use_spectral_norm),
        use_mbstd=str2bool(args.use_mbstd),
        aug_flip=str2bool(args.aug_flip),
        save_intermediate_samples=str2bool(args.save_intermediate_samples),
        save_intermediate_ckpt=str2bool(args.save_intermediate_ckpt),
        sample_every=args.sample_every,
        ckpt_every=args.ckpt_every,
        resume=args.resume,
        seed=args.seed,
        deterministic=str2bool(args.deterministic),
    )

    main(cfg)


# python train_dcgan_64.py ^
#   --data_root C:\Users\Administrator\Desktop\GAN\bellflower ^
#   --outdir .\outputs_dcgan64 ^
#   --epochs 2000 ^
#   --batch_size 128 ^
#   --workers 4 ^ 
#   --save_intermediate_ckpt false ^
#   --save_intermediate_samples true ^
#   --sample_every 200 