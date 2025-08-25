python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install --upgrade pip
# CUDA에 맞는 PyTorch 설치 (예: CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tqdm pillow

python train_dcgan_64.py \
  --data_root /bellflower \
  --outdir ./outputs_dcgan64 \
  --epochs 2000 \
  --batch_size 128 \
  --save_intermediate_ckpt false \
  --save_intermediate_samples false


Epoch 576/2000: 100%|█████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.29, lossG=1.12] 
Epoch 577/2000: 100%|█████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.26, lossG=0.89] 
Epoch 578/2000: 100%|████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.28, lossG=0.842] 
Epoch 579/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.27, lossG=0.877] 
Epoch 580/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.27, lossG=0.836] 
Epoch 581/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.26, lossG=0.825] 
Epoch 582/2000: 100%|████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.27, lossG=0.805] 
Epoch 583/2000: 100%|████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.27, lossG=0.777] 
Epoch 584/2000: 100%|████| 6/6 [00:13<00:00,  2.17s/it, lossD=1.32, lossG=0.633] 
Epoch 585/2000: 100%|████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.27, lossG=0.836] 
Epoch 586/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.26, lossG=0.876] 
Epoch 587/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.29, lossG=0.951] 
Epoch 588/2000: 100%|████████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.25, lossG=1] 
Epoch 589/2000: 100%|█████| 6/6 [00:12<00:00,  2.16s/it, lossD=1.27, lossG=1.06] 
Epoch 590/2000: 100%|████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.27, lossG=0.858] 
Epoch 591/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.27, lossG=0.835] 
Epoch 592/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.26, lossG=0.821] 
Epoch 593/2000: 100%|████| 6/6 [00:13<00:00,  2.18s/it, lossD=1.26, lossG=0.829] 
Epoch 594/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.27, lossG=0.717] 
Epoch 595/2000: 100%|████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.28, lossG=0.763] 
Epoch 596/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.24, lossG=0.897] 
Epoch 597/2000: 100%|█████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.28, lossG=0.91] 
Epoch 598/2000: 100%|█████| 6/6 [00:13<00:00,  2.17s/it, lossD=1.27, lossG=0.99] 
Epoch 599/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.23, lossG=0.874] 
Epoch 600/2000: 100%|█████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.26, lossG=1.01] 
Epoch 601/2000: 100%|████| 6/6 [00:12<00:00,  2.16s/it, lossD=1.27, lossG=0.909] 
Epoch 602/2000: 100%|████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.27, lossG=0.731] 
Epoch 603/2000: 100%|████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.26, lossG=0.749] 
Epoch 604/2000: 100%|████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.26, lossG=0.809] 
Epoch 605/2000: 100%|████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.27, lossG=0.821] 
Epoch 606/2000: 100%|████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.26, lossG=0.872] 
Epoch 607/2000: 100%|█████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.23, lossG=0.87] 
Epoch 608/2000: 100%|████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.29, lossG=0.661] 
Epoch 609/2000: 100%|████| 6/6 [00:12<00:00,  2.16s/it, lossD=1.25, lossG=0.788] 
Epoch 610/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.26, lossG=0.734] 
Epoch 611/2000: 100%|████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.26, lossG=0.844] 
Epoch 612/2000: 100%|████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.24, lossG=0.862] 
Epoch 613/2000: 100%|████████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.23, lossG=1]
Epoch 614/2000: 100%|█████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.23, lossG=1.03] 
Epoch 615/2000: 100%|████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.25, lossG=0.947] 
Epoch 616/2000: 100%|█████| 6/6 [00:12<00:00,  2.16s/it, lossD=1.26, lossG=1.01] 
Epoch 617/2000: 100%|████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.26, lossG=0.945] 
Epoch 618/2000: 100%|█████| 6/6 [00:12<00:00,  2.14s/it, lossD=1.26, lossG=1.02] 
Epoch 619/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.27, lossG=0.987] 
Epoch 620/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.25, lossG=0.909] 
Epoch 621/2000: 100%|████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.21, lossG=0.986] 
Epoch 622/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.22, lossG=0.971] 
Epoch 623/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.24, lossG=0.898] 
Epoch 624/2000: 100%|█████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.24, lossG=1.05] 
Epoch 625/2000: 100%|████| 6/6 [00:12<00:00,  2.10s/it, lossD=1.24, lossG=0.937] 
Epoch 626/2000: 100%|████| 6/6 [00:12<00:00,  2.10s/it, lossD=1.23, lossG=0.913] 
Epoch 627/2000: 100%|████| 6/6 [00:12<00:00,  2.16s/it, lossD=1.26, lossG=0.974] 
Epoch 628/2000: 100%|████| 6/6 [00:12<00:00,  2.13s/it, lossD=1.26, lossG=0.981] 
Epoch 629/2000: 100%|█████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.26, lossG=1.04] 
Epoch 630/2000: 100%|████████| 6/6 [00:12<00:00,  2.11s/it, lossD=1.25, lossG=1] 
Epoch 631/2000: 100%|████| 6/6 [00:12<00:00,  2.12s/it, lossD=1.22, lossG=0.976] 
Epoch 632/2000: 100%|████| 6/6 [00:12<00:00,  2.15s/it, lossD=1.26, lossG=0.955] 
Epoch 633/2000: 100%|████| 6/6 [00:11<00:00,  1.92s/it, lossD=1.25, lossG=0.887] 
Epoch 634/2000: 100%|████| 6/6 [00:10<00:00,  1.80s/it, lossD=1.23, lossG=0.994] 
Epoch 635/2000: 100%|█████| 6/6 [00:10<00:00,  1.83s/it, lossD=1.24, lossG=1.09] 
Epoch 636/2000: 100%|█████| 6/6 [00:10<00:00,  1.79s/it, lossD=1.25, lossG=1.07] 
Epoch 637/2000: 100%|████| 6/6 [00:11<00:00,  1.89s/it, lossD=1.24, lossG=0.902] 
Epoch 638/2000: 100%|████| 6/6 [00:11<00:00,  1.96s/it, lossD=1.24, lossG=0.911] 
Epoch 639/2000: 100%|████| 6/6 [00:13<00:00,  2.19s/it, lossD=1.24, lossG=0.898] 
Epoch 640/2000: 100%|████| 6/6 [00:12<00:00,  2.04s/it, lossD=1.22, lossG=0.856] 
Epoch 641/2000: 100%|█████| 6/6 [00:14<00:00,  2.34s/it, lossD=1.25, lossG=1.01] 
Epoch 642/2000: 100%|█████| 6/6 [00:12<00:00,  2.03s/it, lossD=1.25, lossG=1.03] 
Epoch 643/2000: 100%|█████| 6/6 [00:12<00:00,  2.02s/it, lossD=1.25, lossG=1.07] 
Epoch 644/2000:   0%|                                     | 0/6 [00:00<?, ?it/sT 
raceback (most recent call last):
  File "<string>", line 1, in <module>


python train_dcgan_64.py ^
  --data_root C:\Users\user_\Desktop\flower\bellflower ^
  --outdir .\outputs_dcgan64_cpu ^
  --epochs 2000 ^
  --batch_size 128 ^
  --workers 4 ^
  --save_intermediate_ckpt false ^
  --save_intermediate_samples true ^
  --sample_every 50 ^
  --resume .\outputs_dcgan64\ckpt\best.pt
