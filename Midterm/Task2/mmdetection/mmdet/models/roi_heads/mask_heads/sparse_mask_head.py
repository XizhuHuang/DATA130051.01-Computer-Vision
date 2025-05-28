from mmdet.models.roi_heads.mask_heads.dynamic_mask_head import DynamicMaskHead
from mmdet.registry import MODELS

@MODELS.register_module()
class SparseMaskHead(DynamicMaskHead):
    def __init__(self,
                 num_classes,
                 num_convs=4,
                 conv_out_channels=256, 
                 input_feat_shape=14,
                 in_channels=256,
                 dynamic_conv_cfg=dict(
                     type='DynamicConv',
                     in_channels=256,
                     feat_channels=64,
                     out_channels=256,
                     input_feat_shape=14,
                     act_cfg=dict(type='ReLU', inplace=True),
                     norm_cfg=dict(type='LN')),
                 # 注意：在这里移除 loss_mask 参数
                 **kwargs):  # 捕获额外参数
        # 构建 loss_mask，并加入 kwargs 传入父类
        loss_mask = dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        kwargs.update(dict(
            num_convs=num_convs,
            roi_feat_size=input_feat_shape,
            in_channels=in_channels,
            conv_out_channels=conv_out_channels,
            num_classes=num_classes,
            class_agnostic=False,
            upsample_cfg=dict(type='deconv', scale_factor=2),
            dynamic_conv_cfg=dynamic_conv_cfg,
            loss_mask=loss_mask
        ))

        super().__init__(**kwargs)

    def forward(self, mask_feats, attn_feats):
        return super().forward(mask_feats, attn_feats)
