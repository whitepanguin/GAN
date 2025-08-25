from train_dcgan_64 import generate_from_best

generate_from_best(
    best_path = r".\outputs_dcgan64_cpu\ckpt\best.pt",
    outdir    = r".\outputs_dcgan64_cpu\gen_ema",
    n_images  = 500,
    batch     = 32,
    nz        = 128,
    ngf       = 64,
    nc        = 3,
    prefix    = "ema"
)
