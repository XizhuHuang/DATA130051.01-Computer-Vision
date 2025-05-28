# import os
# import mmcv
# import torch
# from mmdet.apis import init_detector, inference_detector, show_result
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np

# # 模型配置文件和权重路径
# config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc_lowmem.py'
# checkpoint_file = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/epoch_12.pth'

# # 初始化模型（cuda或cpu）
# model = init_detector(config_file, checkpoint_file, device='cuda:0')

# # 你要测试的4张图路径，建议从val或者test集中挑选
# test_img_dir = 'data/VOCdevkit/VOC2007/JPEGImages'
# test_img_names = [
#     '000001.jpg',
#     '000002.jpg',
#     '000004.jpg',
#     '000006.jpg'
# ]

# # 创建保存结果文件夹
# save_dir = 'work_dirs/proposal_results'
# os.makedirs(save_dir, exist_ok=True)

# for img_name in test_img_names:
#     img_path = os.path.join(test_img_dir, img_name)
#     img = mmcv.imread(img_path)
    
#     # 前向推理，得到结果（包括rpn_proposals）
#     with torch.no_grad():
#         # 这里需要用model的forward_test接口（MMDet里没有直接暴露获取proposal的接口）
#         # 简单办法：直接调用inference_detector得到最终结果
#         result = inference_detector(model, img)
        
#         # 如果想拿proposal，需要修改模型forward逻辑或者hook RPN输出
#         # 这里给个简单思路是通过model.rpn_head直接调用
#         # 先准备数据：
#         data = model.data_preprocessor({'img': img}, False)
#         features = model.backbone(data['img'][0])
#         rpn_outs = model.rpn_head(features)
#         # 处理rpn_outs，生成proposal（需调用model.rpn_head.get_bboxes）
#         proposals = model.rpn_head.get_bboxes(*rpn_outs, data['img_metas'][0])
    
#     # 画图：proposal boxes（只画top 100）
#     img_proposal = img.copy()
#     for box in proposals[0][:100]:
#         box = box.int().cpu().numpy()
#         cv2.rectangle(img_proposal, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
#     cv2.imwrite(os.path.join(save_dir, f'{img_name}_proposal.jpg'), img_proposal)
    
#     # 画图：最终预测结果，使用mmdetection自带的可视化函数
#     # model.show_result(img, result, score_thr=0.3, out_file=os.path.join(save_dir, f'{img_name}_final.jpg'))
#     show_result(model, img, result, score_thr=0.3, wait_time=0, show=True, out_file=os.path.join(save_dir, f'{img_name}_final.jpg'))

# print('可视化完成，结果保存在:', save_dir)
########################################################
# import os
# import mmcv
# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# from mmdet.apis import init_detector
# from mmengine.runner import Runner
# from mmengine.structures import InstanceData
# from mmdet.structures import DetDataSample
# from mmdet.registry import VISUALIZERS

# # ✅ 配置路径
# config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc_lowmem.py'
# checkpoint_file = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/epoch_12.pth'

# # ✅ 测试图像路径（请确保存在这些图）
# img_paths = [
#     'data/VOCdevkit/VOC2007/JPEGImages/000001.jpg',
#     'data/VOCdevkit/VOC2007/JPEGImages/000005.jpg',
#     'data/VOCdevkit/VOC2007/JPEGImages/000010.jpg',
#     'data/VOCdevkit/VOC2007/JPEGImages/000015.jpg',
# ]

# # ✅ 保存目录
# save_dir = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/vis_proposal'
# os.makedirs(save_dir, exist_ok=True)

# # ✅ 加载模型
# model = init_detector(config_file, checkpoint_file, device='cuda:0')
# model.eval()

# # ✅ 创建 visualizer 实例
# visualizer = VISUALIZERS.build(model.cfg.visualizer)
# visualizer.dataset_meta = model.dataset_meta

# # ✅ 开始处理图像
# for img_path in img_paths:
#     img_name = os.path.basename(img_path)
#     img = mmcv.imread(img_path)
#     img_tensor = model.data_preprocessor(img, False)['inputs'].to(model.device)

#     with torch.no_grad():
#         # ✅ 提取特征
#         feat = model.backbone(img_tensor)
#         feat = model.neck(feat)

#         # ✅ RPN Proposal boxes
#         rpn_results = model.rpn_head.simple_test_rpn(feat, [img_tensor.shape[2:]])
#         proposals = rpn_results[0][:100]  # 取前100个proposal

#         # ✅ RCNN Final prediction
#         results = model.roi_head.simple_test(
#             feat, rpn_results, [img_tensor.shape[2:]], rescale=True)

#     # ✅ 可视化 RPN proposals
#     img_rpn = img.copy()
#     for bbox in proposals:
#         x1, y1, x2, y2 = bbox.astype(int)
#         cv2.rectangle(img_rpn, (x1, y1), (x2, y2), (255, 255, 0), 1)
#     cv2.putText(img_rpn, 'RPN Proposals', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
#     cv2.imwrite(os.path.join(save_dir, img_name.replace('.jpg', '_proposal.jpg')), img_rpn)

#     # ✅ 可视化 RCNN final results（bbox + mask）
#     data_sample = results[0]
#     drawn_img = visualizer.add_datasample(
#         img_name,
#         img,
#         data_sample,
#         draw_gt=False,
#         show=False,
#         wait_time=0,
#         out_file=os.path.join(save_dir, img_name.replace('.jpg', '_final.jpg'))
#     )

# print(f"✅ 可视化完成，保存于：{save_dir}")
import os
import cv2
import torch
import numpy as np
from mmengine.config import Config
from mmdet.apis import init_detector
from mmengine.registry import VISUALIZERS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


def clip_bboxes(bboxes, img_shape):
    """裁剪边界框坐标到图像范围内
    
    Args:
        bboxes (Tensor): 边界框坐标，形状为 [N, 4]，格式为 (x1, y1, x2, y2)
        img_shape (tuple): 图像尺寸 (height, width)
    
    Returns:
        Tensor: 裁剪后的坐标
    """
    h, w = img_shape[:2]
    bboxes[:, 0::2] = bboxes[:, 0::2].clamp(0, w)  # 限制x坐标在 [0, width]
    bboxes[:, 1::2] = bboxes[:, 1::2].clamp(0, h)  # 限制y坐标在 [0, height]
    return bboxes


# 配置文件和模型路径
config_file = 'configs/mask_rcnn/mask_rcnn_r50_fpn_1x_voc_lowmem.py'
checkpoint_file = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/epoch_24.pth'

# config_file = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_coco.py'
# checkpoint_file = 'work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/epoch_36.pth'

# 初始化模型并获取设备信息
model = init_detector(config_file, checkpoint_file, device='cuda:0')
device = next(model.parameters()).device

# 设置 Visualizer
save_dir = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/vis_final'

# save_dir = 'work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/vis_final'
os.makedirs(save_dir, exist_ok=True)
visualizer = VISUALIZERS.build(model.cfg.visualizer)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

visualizer.dataset_meta = {
    'classes': VOC_CLASSES,
    'palette': [(220, 20, 60)] * len(VOC_CLASSES)  # 可选，颜色设置
}
visualizer.save_dir = save_dir


# 选择4张测试图像路径
img_paths = [
    'data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000045.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000059.jpg',
    'data/VOCdevkit/VOC2007/JPEGImages/000105.jpg',
]

for img_path in img_paths:
    # 读取图像并转换到张量
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 创建带有元数据的数据样本
    data_sample = DetDataSample()
    data_sample.set_metainfo({
        'img_shape': (h, w),
        'ori_shape': (h, w),
        'scale_factor': (1.0, 1.0),
        'img_path': img_path
    })
    
    # 构造输入数据格式
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
    data = {
        'inputs': [img_tensor],
        'data_samples': [data_sample]
    }
    
    # 数据预处理
    processed_data = model.data_preprocessor(data, False)
    batch_inputs = processed_data['inputs']
    batch_data_samples = processed_data['data_samples']

    
    with torch.no_grad():
        feats = model.extract_feat(batch_inputs)
        proposal_list = model.rpn_head.predict(feats, batch_data_samples, rescale=False)
        
        print(f"[{os.path.basename(img_path)}] Proposals: {proposal_list[0].bboxes.shape}")
        results = model.roi_head.predict(feats, proposal_list, batch_data_samples)
    
    # 构造可视化数据结构
    det_sample = batch_data_samples[0].clone()
    det_sample.pred_instances = results[0]
    det_sample.proposals = InstanceData(bboxes=proposal_list[0].bboxes)

    # 检查数据结构的完整性
    print("Proposals是否有效:", hasattr(det_sample, 'proposals'))
    print("Proposals是否包含bboxes:", 'bboxes' in det_sample.proposals)

    
    # 可视化处理
    visual_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 可视化最终预测（自动显示预测结果）
    visualizer.add_datasample(
        name=os.path.basename(img_path).replace('.jpg', '_prediction'),
        image=visual_img,
        data_sample=det_sample,
        draw_gt=False,
        show=False,
        pred_score_thr=0.25,
        out_file=os.path.join(save_dir, os.path.basename(img_path).replace('.jpg', '_prediction.jpg'))
    )



print(f"可视化结果已保存至 {save_dir}")