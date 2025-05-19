from ultralytics import YOLO

# 1. 載入您之前訓練的模型權重 (進行微調)
# << 從上次的最佳權重開始微調
model = YOLO("VDL_best_2347.pt")

# 2. 設定訓練參數
results = model.train(
    data="GlobalWheat2020pseudo.yaml",  # << 使用新的 YAML 檔案
    epochs=80,               # << 微調時的 epoch 數量可以適當減少，例如 50-100
    imgsz=1024,
    batch=8,
    device=0,
    workers=8,
    optimizer="SGD",
    lr0=0.001,  lrf=0.0008,    # << 微調時，初始學習率 (lr0) 通常需要調低
    # lrf 也可相應調整或保持較小值
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    mosaic=0.5,  # 在微調初期可以考慮降低 mosaic 的強度或關閉，後期再開啟
    mixup=0.1,  # MixUp 機率也可以適當調整
    degrees=10, translate=0.1, scale=0.5, shear=2,
    flipud=0.0, fliplr=0.5,
    cos_lr=True,
    amp=True,
    # 紀錄＋最佳化
    project="runs/train_pseudo",  # << 新的 project 名稱，避免覆蓋
    name="yolov11x_gwhd1024_pseudo_ft",  # << 新的 run 名稱
    exist_ok=False,  # 確保不會覆蓋之前的實驗
    # resume=False # 如果是微調，通常不 resume，而是載入權重開始新訓練
)
