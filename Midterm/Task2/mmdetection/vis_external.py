import os
from mmengine.config import Config
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
import mmcv
import cv2  # 新增，用于颜色空间转换

# 模型配置和权重路径
config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc_lowmem.py'
checkpoint_file = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/epoch_24.pth'

# config_file = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_coco.py'
# checkpoint_file = 'work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/epoch_36.pth'

# 初始化模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 读取三张外部图片
img_dir = 'demo/external_voc_images'
img_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']
img_paths = [os.path.join(img_dir, fname) for fname in img_list]

# 确认文件存在
for img_path in img_paths:
    print(f"Checking: {img_path}, Exists: {os.path.exists(img_path)}")

# 可视化器初始化
visualizer = DetLocalVisualizer()
visualizer.dataset_meta = model.dataset_meta

# 检测并可视化
for img_path in img_paths:
    result = inference_detector(model, img_path)
    # 读取BGR图像并转换为RGB
    image_bgr = mmcv.imread(img_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    visualizer.add_datasample(
        name=os.path.basename(img_path),
        image=image_rgb,  # 使用转换后的RGB图像
        data_sample=result,
        draw_gt=False,
        show=False,  # 不显示窗口
        # pred_score_thr=0.5465,
        pred_score_thr=0.25,
        out_file=f'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/external_results/{os.path.basename(img_path)}'
        # out_file=f'work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/external_results/{os.path.basename(img_path)}'
    )
