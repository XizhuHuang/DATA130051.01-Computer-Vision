_base_ = [
    '../_base_/models/sparse_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]

# VOC 20 类定义
VOC_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'  # 根据你数据路径调整

# 图像处理流水线（适配显存限制）
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='data/coco/voc07_train.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root=data_root,
        pipeline=train_pipeline,
        metainfo=dict(classes=VOC_classes)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='data/coco/voc07_val.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=dict(classes=VOC_classes)
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='data/coco/voc07_test.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root=data_root,
        pipeline=test_pipeline,
        metainfo=dict(classes=VOC_classes)
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/voc07_val.json',
    metric=['bbox', 'segm']
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/voc07_test.json',
    metric=['bbox', 'segm']
)

# ✅ 修改 num_classes 并显式加上 type，避免构建失败
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(type='DIIHead', num_classes=20)
            for _ in range(6)
        ],
        mask_head=dict(
            type='SparseMaskHead',
            num_classes=20,
            input_feat_shape=14,  # 确保和配置一致
            # 其他参数可根据需要调整
        ),
    ),
    test_cfg=dict(
        rpn=None,
        rcnn=dict(
            max_per_img=100,
            mask_thr_binary=0.5
        )
    )
)

# 优化器设置（AdamW + 梯度裁剪）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001))

# TensorBoard 可视化
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

# 日志设置
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50)
)

# ✅ 可选：加载 COCO 预训练模型进行 finetune（建议）
# load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_3x_coco/sparse_rcnn_r50_fpn_3x_coco_20220526_002856-9048df19.pth'
