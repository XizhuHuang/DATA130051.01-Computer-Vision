# _base_ = [
#     '../_base_/models/mask-rcnn_r50_fpn.py',
#     '../_base_/datasets/coco_instance.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
# ]
# 在 mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py 基础上修改
# 在 mmdetection/configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py 基础上修改

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# 1. VOC类别定义
voc_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# 2. 覆盖数据集路径配置
data = dict(
    samples_per_gpu=1,  # 根据GPU显存调整
    workers_per_gpu=2,
    train=dict(
        type='CocoDataset',
        ann_file='data/coco/voc07_train.json',
        img_prefix='data/VOCdevkit/VOC2007/JPEGImages/',
        classes=voc_classes
    ),
    val=dict(
        type='CocoDataset',
        ann_file='data/coco/voc07_val.json',
        img_prefix='data/VOCdevkit/VOC2007/JPEGImages/',
        classes=voc_classes
    ),
    test=dict(
        type='CocoDataset',
        ann_file='data/coco/voc07_test.json',
        img_prefix='data/VOCdevkit/VOC2007/JPEGImages/',
        classes=voc_classes
    )
)

# 3. 覆盖数据加载器配置（关键路径修正）
train_dataloader = dict(
    dataset=dict(
        ann_file='data/coco/voc07_train.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root='data/VOCdevkit/'
    )
)

val_dataloader = dict(
    dataset=dict(
        ann_file='data/coco/voc07_val.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root='data/VOCdevkit/'
    )
)

test_dataloader = dict(
    dataset=dict(
        ann_file='data/coco/voc07_test.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root='data/VOCdevkit/'
    )
)

# 4. 修正评估器路径
val_evaluator = dict(
    ann_file='data/coco/voc07_val.json',  # 直接使用完整路径
    metric=['bbox', 'segm'],
    type='CocoMetric'
)

test_evaluator = dict(
    ann_file='data/coco/voc07_test.json',
    metric=['bbox', 'segm'],
    type='CocoMetric'
)

# 5. 模型头部类别数调整
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),  # COCO默认80类 → VOC的20类
        mask_head=dict(num_classes=20)
    )
)

# 6. 优化器调整（单GPU学习率）
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)

# # 7. 预训练权重配置（可选）
# load_from = 'checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'  # 若需加载COCO预训练模型


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]

train_dataloader['dataset']['pipeline'] = train_pipeline
val_dataloader['dataset']['pipeline'] = test_pipeline
test_dataloader['dataset']['pipeline'] = test_pipeline

