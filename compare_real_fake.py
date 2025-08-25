import os, torch
from torchvision import datasets, transforms
import torchvision.utils as vutils
from train_dcgan_64 import load_generator_from_best

def compare_real_fake(best_path, data_root, outdir, n_show=16, image_size=64, nz=128, ngf=64, nc=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- 데이터셋 (real) -----------------
    tfms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*nc, [0.5]*nc),
    ])
    ds = datasets.ImageFolder(root=data_root, transform=tfms)
    real_loader = torch.utils.data.DataLoader(ds, batch_size=n_show, shuffle=True)
    real_batch, _ = next(iter(real_loader))

    # ----------------- Generator (fake) -----------------
    G = load_generator_from_best(best_path, device, nz, ngf, nc)
    with torch.no_grad():
        z = torch.randn(n_show, nz, 1, 1, device=device)
        fake_batch = G(z).cpu()

    # ----------------- 저장 -----------------
    os.makedirs(outdir, exist_ok=True)
    grid_real = vutils.make_grid(real_batch, nrow=int(n_show**0.5), normalize=True, value_range=(-1,1))
    grid_fake = vutils.make_grid(fake_batch, nrow=int(n_show**0.5), normalize=True, value_range=(-1,1))

    vutils.save_image(grid_real, os.path.join(outdir, "real_grid.png"))
    vutils.save_image(grid_fake, os.path.join(outdir, "fake_grid.png"))

    # real + fake 비교 grid (위=real, 아래=fake)
    both = torch.cat([grid_real, grid_fake], dim=1)  # H*2 x W
    vutils.save_image(both, os.path.join(outdir, "compare_real_fake.png"))

    print(f"[Saved] real_grid.png, fake_grid.png, compare_real_fake.png in {outdir}")

if __name__ == "__main__":
    compare_real_fake(
        best_path = r".\outputs_dcgan64_cpu\ckpt\best.pt",
        data_root = r"C:\Users\user_\Desktop\flower\bellflower",
        outdir    = r".\compare_out",
        n_show    = 16,
        image_size=64
    )
