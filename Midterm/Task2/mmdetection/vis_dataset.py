import os
import json
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# ======= 可修改的路径 =======
json_file = 'data/coco/voc07_train.json'
image_dir = 'data/VOCdevkit/VOC2007/JPEGImages'
save_path = 'voc2007_dataset_visualization.png'
image_ids_to_show = [12,17,23,26]
# ============================

def draw_bbox(image, anns, cat_id_to_name):
    for ann in anns:
        bbox = ann['bbox']
        cat_id = ann['category_id']
        label = cat_id_to_name[cat_id]

        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    return image

def visualize_voc_images(json_file, image_dir, save_path, image_ids):
    coco = COCO(json_file)

    cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for i, img_id in enumerate(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 原图（第1行）
        axes[0, i].imshow(img_rgb)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Image ID: {img_id}")

        # 标注图（第2行）
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_with_bbox = draw_bbox(img_rgb.copy(), anns, cat_id_to_name)

        axes[1, i].imshow(img_with_bbox)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Labeled Image {i+1}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"✅ Saved to {save_path}")

# 调用函数
visualize_voc_images(json_file, image_dir, save_path, image_ids_to_show)
