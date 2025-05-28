import os
import json
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

# VOC类别列表（共20类）
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def parse_args():
    parser = argparse.ArgumentParser(description='Convert VOC to COCO format')
    parser.add_argument('--voc_path', default='data/VOCdevkit', help='VOC数据集根目录')
    parser.add_argument('--output_dir', default='data/coco', help='COCO格式输出目录')
    return parser.parse_args()

def convert(voc_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        # 创建COCO数据结构
        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i+1, "name": cls} for i, cls in enumerate(VOC_CLASSES)]
        }

        # 读取VOC的ImageSet文件（例如train.txt）
        imgset_path = os.path.join(voc_path, 'VOC2007/ImageSets/Main', f'{split}.txt')
        with open(imgset_path, 'r') as f:
            img_ids = [line.strip() for line in f]

        ann_id = 1
        for img_id in tqdm(img_ids, desc=f'Processing {split}'):
            # 解析XML标注文件
            xml_path = os.path.join(voc_path, 'VOC2007/Annotations', f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 提取图像信息
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image_info = {
                "id": int(img_id),
                "file_name": f'{img_id}.jpg',
                "width": width,
                "height": height
            }
            coco["images"].append(image_info)

            # 提取标注信息
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                w = xmax - xmin
                h = ymax - ymin

                # COCO标注格式（注意：坐标从0开始）
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": int(img_id),
                    "category_id": VOC_CLASSES.index(name) + 1,
                    "bbox": [xmin, ymin, w, h],
                    "area": w * h,
                    "iscrowd": 0
                })
                ann_id += 1

        # 保存为JSON文件
        output_path = os.path.join(output_dir, f'voc07_{split}.json')
        with open(output_path, 'w') as f:
            json.dump(coco, f)
        print(f'Saved COCO format annotations to {output_path}')

if __name__ == '__main__':
    args = parse_args()
    convert(args.voc_path, args.output_dir)