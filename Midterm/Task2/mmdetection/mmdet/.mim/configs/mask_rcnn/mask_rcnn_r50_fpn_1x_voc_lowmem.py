# _base_ = [
#     '../_base_/models/mask-rcnn_r50_fpn.py',
#     '../_base_/schedules/schedule_2x.py',
#     '../_base_/default_runtime.py'
# ]

# # VOC 20 类
# VOC_classes = [
#     'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#     'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
#     'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
# ]

# # 数据路径和格式
# dataset_type = 'CocoDataset'
# data_root = 'data/VOCdevkit/'

# # 数据预处理：缩小图像尺寸以减少显存消耗
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', scale=(640, 480), keep_ratio=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PackDetInputs')
# ]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(640, 480), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='PackDetInputs')
# ]

# train_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         ann_file='data/coco/voc07_train.json',
#         data_prefix=dict(img='VOC2007/JPEGImages/'),
#         data_root='data/VOCdevkit/',
#         pipeline=train_pipeline,
#         metainfo=dict(classes=VOC_classes)
#     )
# )


# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         ann_file='data/coco/voc07_val.json',
#         data_prefix=dict(img='VOC2007/JPEGImages/'),
#         data_root='data/VOCdevkit/',
#         pipeline=test_pipeline,
#         metainfo=dict(classes=VOC_classes)
#     )
# )

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     dataset=dict(
#         type=dataset_type,
#         ann_file='data/coco/voc07_test.json',
#         data_prefix=dict(img='VOC2007/JPEGImages/'),
#         data_root='data/VOCdevkit/',
#         pipeline=test_pipeline,
#         metainfo=dict(classes=VOC_classes)
#     )
# )





# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file='data/coco/voc07_val.json',
#     metric=['bbox', 'segm']
# )

# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file='data/coco/voc07_test.json',
#     metric=['bbox', 'segm']
# )


# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=20),
#         mask_head=dict(num_classes=20)
#     ),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.7),
#             min_bbox_size=0,
#         ),
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100
#         )
#     )
# )



# # 可选：启用混合精度（进一步省内存）
# # fp16 = dict(loss_scale=512.)


# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))

# # default_hooks = dict(
# #     logger=dict(type='LoggerHook', interval=50),
# #     visualization=dict(type='TensorboardVisHook')  # ✅ 启用 TensorBoard 可视化
# # )
# # 可视化后端配置（关键修正）
# vis_backends = [
#     dict(type='LocalVisBackend'),  # 本地保存可视化结果
#     dict(type='TensorboardVisBackend')  # 关键：启用TensorBoard
# ]
# visualizer = dict(
#     type='DetLocalVisualizer',
#     vis_backends=vis_backends,
#     name='visualizer' 
# )
# # 日志配置
# default_hooks = dict(
#     logger=dict(type='LoggerHook', interval=50)
# )

_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

# VOC 20 类
VOC_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

dataset_type = 'CocoDataset'
data_root = 'data/VOCdevkit/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),  
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        ann_file='data/coco/voc07_train.json',
        data_prefix=dict(img='VOC2007/JPEGImages/'),
        data_root='data/VOCdevkit/',
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
        data_root='data/VOCdevkit/',
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
        data_root='data/VOCdevkit/',
        pipeline=test_pipeline,
        metainfo=dict(classes=VOC_classes)
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/voc07_val.json',
    metric=['bbox']  
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/voc07_test.json',
    metric=['bbox']  
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=None 
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend')
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50)
)
