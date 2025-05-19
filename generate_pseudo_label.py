from ultralytics import YOLO
import os
import glob
import torch  # << 匯入 torch

# 0. 設定參數
MODEL_PATH = "runs/train/yolov11x_gwhd1024_phase2/weights/best.pt"
CONF_THRESHOLD = 0.5
IMG_SIZE_PREDICT = 1024  # << 推斷時的圖片尺寸，如果OOM可以先嘗試調低這個值
DEVICE = 0
AUGMENT = True

TEST_IMAGE_DIRS = [
    "datasets/GlobalWheat2020/images/utokyo_1",
    "datasets/GlobalWheat2020/images/utokyo_2",
    "datasets/GlobalWheat2020/images/uq_1",
    "datasets/GlobalWheat2020/images/nau_1"
]

for dir_path in TEST_IMAGE_DIRS:

    PSEUDO_LABEL_OUTPUT_PROJECT = "runs/pseudo_labeling"
    PSEUDO_LABEL_OUTPUT_NAME = os.path.basename(dir_path)

    # 1. 載入您訓練好的模型
    model = YOLO(MODEL_PATH)

    # 2. 收集所有測試圖片路徑
    all_test_images = []
    for dir_path in TEST_IMAGE_DIRS:
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            all_test_images.extend(glob.glob(os.path.join(dir_path, ext)))

    print(f"找到 {len(all_test_images)} 張測試圖片進行偽標籤。")
    print(f"將使用 imgsz={IMG_SIZE_PREDICT} 進行預測。")

    # 3. 逐張圖片進行預測並儲存 YOLO 格式的標籤檔案
    #    確保 project 和 name 參數能讓 predict 函式將標籤儲存在同一個 'labels' 資料夾下
    processed_count = 0
    for i, image_path in enumerate(all_test_images):
        print(f"正在處理圖片 {i+1}/{len(all_test_images)}: {image_path}")
        try:
            model.predict(
                source=image_path,       # 一次處理一張圖片
                imgsz=IMG_SIZE_PREDICT,
                conf=CONF_THRESHOLD,
                save_txt=True,
                save_conf=False,
                project=PSEUDO_LABEL_OUTPUT_PROJECT,
                name=PSEUDO_LABEL_OUTPUT_NAME,
                augment=AUGMENT,
                exist_ok=True,         # 非常重要，確保不會覆蓋已生成的標籤
                device=DEVICE,
                verbose=False,         # 在迴圈中可以關閉詳細輸出，除非需要偵錯
                half=False              # 明確啟用半精度
            )
            processed_count += 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"警告：處理圖片 {image_path} 時發生 CUDA OOM 錯誤。")
                print("嘗試清理 CUDA 快取後繼續...")
                torch.cuda.empty_cache()  # 清理快取
                # 您可以選擇跳過這張圖片，或者在這裡嘗試用更小的 imgsz 重試一次
                # 例如：嘗試用更小的尺寸重試一次
                # try:
                #     print(f"嘗試使用更小的尺寸 (640) 重試圖片: {image_path}")
                #     model.predict(source=image_path, imgsz=640, ..., exist_ok=True) # 其他參數保持一致
                #     processed_count +=1
                # except RuntimeError as e2:
                #     if "out of memory" in str(e2).lower():
                #         print(f"使用小尺寸重試圖片 {image_path} 仍然 OOM，跳過此圖片。")
                #     else: raise e2 # 其他錯誤則拋出
                print(f"跳過圖片 {image_path}。")
                continue  # 繼續處理下一張圖片
            else:
                raise e  # 如果不是OOM錯誤，則重新拋出

        # 定期清理 CUDA 快取 (例如每 10 或 20 張圖片)
        if (i + 1) % 10 == 0:
            print(f"已處理 {i+1} 張圖片，清理 CUDA 快取...")
            torch.cuda.empty_cache()

    print(f"偽標籤生成完成。總共成功處理 {processed_count} / {len(all_test_images)} 張圖片。")
    print(
        f"偽標籤已儲存在: {os.path.join(PSEUDO_LABEL_OUTPUT_PROJECT, PSEUDO_LABEL_OUTPUT_NAME, 'labels')}")

# 最後再清理一次
torch.cuda.empty_cache()
