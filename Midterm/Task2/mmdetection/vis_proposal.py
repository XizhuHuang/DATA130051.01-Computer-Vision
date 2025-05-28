import os
import cv2
import torch
import numpy as np
from mmdet.apis import init_detector
from mmdet.structures import DetDataSample

# 配置文件和模型权重
config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc_lowmem.py'
checkpoint_file = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/epoch_24.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')
device = next(model.parameters()).device

# 输出目录
save_dir = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/vis_proposal'
os.makedirs(save_dir, exist_ok=True)

# 选取 4 张图像
img_paths = [
    'data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000045.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000059.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000105.jpg',
]

for img_path in img_paths:
    # 读取图像
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # 构建 DataSample 包含必要元信息
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape': (h, w),
        'ori_shape': (h, w),
        'scale_factor': (1.0, 1.0),
        'img_path': img_path
    })

    # 准备输入张量
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    data = {
        'inputs': [img_tensor],
        'data_samples': [data_sample]
    }

    # 数据预处理
    processed = model.data_preprocessor(data, False)
    inputs = processed['inputs']
    data_samples = processed['data_samples']

    with torch.no_grad():
        feats = model.extract_feat(inputs)
        proposal_list = model.rpn_head.predict(feats, data_samples, rescale=False)

    proposals = proposal_list[0].bboxes.cpu().numpy()
    proposals = proposals[:50]


    # 画图：每个proposal画一个框
    for bbox in proposals:
        x1, y1, x2, y2 = bbox.astype(int)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 165, 0), thickness=1)

    # 保存图像
    out_path = os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '_rpn_proposals.jpg'))
    cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved proposal visualization: {out_path}")

print(f"\n所有 proposal 可视化图像已保存至：{save_dir}")
