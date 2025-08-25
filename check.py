# check.py (OOM-safe)
import os, argparse, torch, torch.nn as nn
import torchvision.utils as vutils
from torchvision import datasets, transforms

def tstats(t, name):
    t = t.detach().float().cpu()
    print(f"[{name}] shape={tuple(t.shape)} min={t.min():.3f} max={t.max():.3f} "
          f"mean={t.mean():.3f} std={t.std():.3f}")

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
            nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

def save_grid01(t, path, nrow=8):
    t = t.detach().cpu().add(1).div(2).clamp_(0,1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(t, path, nrow=nrow)
    print("[saved]", path)

def load_generators(best_path, device):
    # ★ CPU로 먼저 로드해 GPU OOM 회피 (+ 안전 모드)
    try:
        ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(best_path, map_location="cpu")  # (구버전 호환)
    cfg  = ckpt.get("cfg", {})
    nz   = int(cfg.get("nz", 128)); ngf = int(cfg.get("ngf", 64)); nc = int(cfg.get("nc", 3))
    print(f"[ckpt cfg] nz={nz}, ngf={ngf}, nc={nc}")
    print(f"[ckpt meta] epoch={ckpt.get('epoch','?')}, step={ckpt.get('global_step','?')}, best_lossG={ckpt.get('lossG','?')}")

    G_net = Generator(nz, ngf, nc)  # 아직 CPU
    assert "netG" in ckpt, "checkpoint에 'netG'가 없습니다."
    G_net.load_state_dict(ckpt["netG"], strict=True)
    G_net = G_net.to(device)  # 여기서만 GPU 이동

    G_ema, has_ema = None, False
    if ckpt.get("ema_G", None) is not None:
        G_ema = Generator(nz, ngf, nc)
        G_ema.load_state_dict(ckpt["ema_G"], strict=True)
        G_ema = G_ema.to(device)
        has_ema = True
    return G_net, G_ema, has_ema, nz, ngf, nc

def sync_bn_from(A: nn.Module, B: nn.Module):
    with torch.no_grad():
        for bA, bB in zip(A.buffers(), B.buffers()):
            bA.copy_(bB)
        for mA, mB in zip(A.modules(), B.modules()):
            if isinstance(mA, nn.BatchNorm2d) and isinstance(mB, nn.BatchNorm2d):
                mA.weight.copy_(mB.weight); mA.bias.copy_(mB.bias)

def dump_bn_affine(model, tag):
    print(f"\n==== BN AFFINE [{tag}] ====")
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            w = m.weight.detach().float().cpu(); b = m.bias.detach().float().cpu()
            print(f"{n:40s} w_mean={w.mean():.4f} w_std={w.std():.4f} b_mean={b.mean():.4f} b_std={b.std():.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", required=True, type=str)
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--outdir", type=str, default="./check_out")
    ap.add_argument("--mode", type=str, default="both", choices=["eval","train","both"])
    ap.add_argument("--sync_bn", type=str, default="True", choices=["True","False"])
    ap.add_argument("--batch", type=int, default=32)  # 기본 32
    ap.add_argument("--cpu", action="store_true", help="강제로 CPU에서 실행")
    args = ap.parse_args()

    # ★ CPU 옵션 반영
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Device]", device)

    G_net, G_ema, has_ema, nz, ngf, nc = load_generators(args.best, device)
    if has_ema and args.sync_bn == "True":
        sync_bn_from(G_ema, G_net)
        print("[info] ema_G BN buffers + affine synced from netG")
        dump_bn_affine(G_ema, "ema (after sync)")

    # (옵션) real 시각화
    if args.data_root:
        tfms = transforms.Compose([
            transforms.Resize(64), transforms.CenterCrop(64),
            transforms.ToTensor(), transforms.Normalize([0.5]*nc, [0.5]*nc),
        ])
        ds = datasets.ImageFolder(root=args.data_root, transform=tfms)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)
        real, _ = next(iter(dl))
        tstats(real, "real")
        save_grid01(real[:min(64, args.batch)],
                    os.path.join(args.outdir, "real_grid.png"))

    def sample(G: nn.Module, tag: str, train_mode=False):
        G.train(mode=train_mode)
        with torch.no_grad():
            z = torch.randn(args.batch, nz, 1, 1, device=device)
            fake = G(z)
        tstats(fake, f"fake[{tag}][{'train' if train_mode else 'eval'}]")
        save_grid01(fake[:min(64, args.batch)],
                    os.path.join(args.outdir, f"{tag}_{'train' if train_mode else 'eval'}.png"))

    if args.mode in ("eval", "both"):
        sample(G_net.eval(), "netG", train_mode=False)
        if has_ema:
            sample(G_ema.eval(), "ema", train_mode=False)
    if args.mode in ("train", "both"):
        sample(G_net, "netG", train_mode=True)
        if has_ema:
            sample(G_ema, "ema", train_mode=True)


if __name__ == "__main__":
    main()

# python check.py --best "C:\Users\Administrator\Desktop\GAN\outputs_dcgan64\ckpt\best.pt" ^
#   --outdir ".\check_out" ^
#   --mode eval ^
#   --batch 16 ^
#   --sync_bn True ^
#   --cpu

