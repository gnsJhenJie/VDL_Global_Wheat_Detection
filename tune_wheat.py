from ultralytics import YOLO

# 1. 讀取上一次訓練的「最後」檔 (last.pt)，它包含 model、optimizer、lr‐scheduler 等狀態
model = YOLO("runs/train/yolov11x_gwhd1024_new/weights/last.pt")

# 2. 用 resume=True 完整接續訓練
results = model.train(
    data="GlobalWheat2020.yaml",
    # 只要把 epochs 設為你想要再加跑的總迭代數（例如再跑 80）
    epochs=80,
    imgsz=1024,
    batch=8,
    optimizer="AdamW",
    lr0=0.0001, lrf=1e-5,
    mosaic=0.5, mixup=0.1, copy_paste=0.2, close_mosaic=20,
    amp=True, cos_lr=True,
    device=0, workers=8,
    # 如要覆寫同一個 run（強制覆蓋），可以打
    exist_ok=False,
    # name/project 可視需求改或留空（留空會續寫到原 run 資料夾）
    name="yolov11x_gwhd1024_fintune2",
)
