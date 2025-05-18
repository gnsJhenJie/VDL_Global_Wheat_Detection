"""
train_wheat.py  –  RTX 4090 單卡 24GB 最佳化設定
"""
from ultralytics import YOLO

# 1. 載入 YOLOv8-X 預訓練權重（COCO）
model = YOLO("yolo11x.pt")  # 99 M params，大 GPU 能吃

# 2. 設定高解析度 & 強化增強
results = model.train(
    data="GlobalWheat2020.yaml",
    epochs=150,              # 大概 12~14 小時可訓練完
    imgsz=1024,              # 主輸入尺寸（會隨機多尺度）
    batch=8,                 # 4090 24GB 實測 8×1024×1024 剛好
    device=0,                # 指定 GPU
    workers=8,               # 依 CPU 核心數調整
    optimizer="SGD",         # 動量 0.937 / weight_decay 5e‑4 皆預設
    lr0=0.01, lrf=0.01,      # 初始 & 最低 LR（可用 One‑Cycle，預設會自動）
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # 色彩抖動
    mosaic=1.0,              # Mosaic 機率 100 %
    mixup=0.2,               # MixUp 機率 20 %
    degrees=10, translate=0.1, scale=0.5, shear=2,
    flipud=0.0, fliplr=0.5,
    cos_lr=True,             # 餘弦退火
    amp=True,                # 自動混合精度
    # 多尺度：YOLOv8 自帶 random_resize；這裡開啟範圍 0.8–1.2
    # scale_uv=0.2,
    # 紀錄＋最佳化
    project="runs/train",
    name="yolov11x_gwhd1024",
    # loggers="tensorboard",
    exist_ok=False,
)
