import os
from PIL import Image
from tqdm import tqdm

# ================= 配置路径 =================
# 你的原始图片文件夹
INPUT_DIR = "/home/ysc/3dgs-npy/data/taijie/images"

# 你想保存的新文件夹
OUTPUT_DIR = "/home/ysc/3dgs-npy/data/taijie_1600/images"

# 目标尺寸
TARGET_SIZE = (1600, 1600)
# ===========================================

def process_images():
    # 1. 如果输出文件夹不存在，自动创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建新文件夹: {OUTPUT_DIR}")

    # 2. 获取所有图片文件
    valid_extensions = ('.jpg', '.jpeg', '.png', '.JPG', '.PNG')
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith(valid_extensions)]
    
    print(f"找到 {len(files)} 张图片，准备缩放至 {TARGET_SIZE}...")

    # 3. 循环处理
    for filename in tqdm(files):
        img_path = os.path.join(INPUT_DIR, filename)
        save_path = os.path.join(OUTPUT_DIR, filename)

        try:
            with Image.open(img_path) as img:
                # 使用高质量重采样算法 (LANCZOS)
                img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                img_resized.save(save_path, quality=95)
        except Exception as e:
            print(f"处理 {filename} 时出错: {e}")

    print("✅ 全部完成！")
    print(f"新图片保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_images()