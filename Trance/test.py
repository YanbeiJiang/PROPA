import os
import json
import shutil

# ===== 在这里列出所有数据集 =====
datasets = [
    {
        "json_path": "/data/projects/punim1996/Data/CVPR2026_public/Trance/Trance_train.json",
        "dst_root": "../images/train_images/Spatial-Transformation-Full_new"
    },
    {
        "json_path": "/data/projects/punim1996/Data/CVPR2026_public/Trance/Trance_val.json",
        "dst_root": "../images/train_images/Spatial-Transformation-Full_new"
    }
]

# ===== 函数：从 JSON 中复制 images =====
def copy_images_from_json(json_path, dst_root):
    os.makedirs(dst_root, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    copied = set()
    missing = []

    for entry in data:
        image_paths = entry.get("images", [])
        for img_path in image_paths:
            src_path = img_path
            file_name = os.path.basename(src_path)
            dst_path = os.path.join(dst_root, file_name)

            if os.path.exists(src_path):
                if file_name not in copied:
                    shutil.copy2(src_path, dst_path)
                    copied.add(file_name)
            else:
                missing.append(src_path)

    print(f"\n✅ {os.path.basename(json_path)} done!")
    print(f"   → Copied {len(copied)} unique images to {dst_root}")
    if missing:
        print(f"   ⚠️ Missing {len(missing)} images (check paths if needed)")
    return copied, missing


# ===== 批量执行所有数据集 =====
total_copied = 0
for ds in datasets:
    copied, missing = copy_images_from_json(ds["json_path"], ds["dst_root"])
    total_copied += len(copied)

print(f"\n🎉 All datasets processed! Total {total_copied} unique images copied.")
