# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def load_metrics_from_single_json(json_path):
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     return {
#         'bbox_mAP': data.get('coco/bbox_mAP', 0),
#         'bbox_mAP_50': data.get('coco/bbox_mAP_50', 0),
#         'bbox_mAP_75': data.get('coco/bbox_mAP_75', 0),
#         'bbox_mAP_s': data.get('coco/bbox_mAP_s', 0),
#         'bbox_mAP_m': data.get('coco/bbox_mAP_m', 0),
#         'bbox_mAP_l': data.get('coco/bbox_mAP_l', 0),
#         'segm_mAP': data.get('coco/segm_mAP', 0),
#         'segm_mAP_50': data.get('coco/segm_mAP_50', 0),
#         'segm_mAP_75': data.get('coco/segm_mAP_75', 0),
#         'segm_mAP_s': data.get('coco/segm_mAP_s', 0),
#         'segm_mAP_m': data.get('coco/segm_mAP_m', 0),
#         'segm_mAP_l': data.get('coco/segm_mAP_l', 0),
#     }

# def plot_and_save_metrics(json_path, save_dir, save_name='mask_rcnn_eval_plot.png'):
#     metrics = load_metrics_from_single_json(json_path)

#     categories = ['Overall', 'Small', 'Medium', 'Large']
#     bbox_scores = [metrics['bbox_mAP'], metrics['bbox_mAP_s'], metrics['bbox_mAP_m'], metrics['bbox_mAP_l']]
#     segm_scores = [metrics['segm_mAP'], metrics['segm_mAP_s'], metrics['segm_mAP_m'], metrics['segm_mAP_l']]

#     x = np.arange(len(categories))
#     width = 0.35

#     fig, ax = plt.subplots(figsize=(8, 6))
#     bars1 = ax.bar(x - width / 2, bbox_scores, width, label='BBox mAP', color='skyblue')
#     bars2 = ax.bar(x + width / 2, segm_scores, width, label='Segm mAP', color='lightcoral')

#     def add_labels(bars):
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(f'{height:.3f}',
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 3),
#                         textcoords="offset points",
#                         ha='center', va='bottom')

#     add_labels(bars1)
#     add_labels(bars2)

#     ax.set_ylabel('mAP')
#     ax.set_title('COCO-style Evaluation: Mask R-CNN')
#     ax.set_xticks(x)
#     ax.set_xticklabels(categories)
#     ax.set_ylim(0, 0.5)
#     ax.legend()
#     plt.grid(True, linestyle='--', alpha=0.3)
#     plt.tight_layout()

#     # 保存图像
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, save_name)
#     plt.savefig(save_path)
#     print(f"图像已保存到：{save_path}")

#     plt.close()

# # 调用示例
# json_path = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/20250526_110948_test/20250526_110948.json'
# save_dir = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/20250526_110948_test'
# plot_and_save_metrics(json_path, save_dir)

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def load_metrics_from_single_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    return {
        'bbox_mAP': data.get('coco/bbox_mAP', 0),
        'bbox_mAP_50': data.get('coco/bbox_mAP_50', 0),
        'bbox_mAP_75': data.get('coco/bbox_mAP_75', 0),
        'bbox_mAP_s': data.get('coco/bbox_mAP_s', 0),
        'bbox_mAP_m': data.get('coco/bbox_mAP_m', 0),
        'bbox_mAP_l': data.get('coco/bbox_mAP_l', 0),
    }

def plot_model_comparison(mask_json, sparse_json, save_dir, save_name='model_comparison_eval_plot.png'):
    mask_metrics = load_metrics_from_single_json(mask_json)
    sparse_metrics = load_metrics_from_single_json(sparse_json)

    categories = ['Overall', 'mAP@50', 'mAP@75', 'Small', 'Medium', 'Large']
    mask_scores = [
        mask_metrics['bbox_mAP'],
        mask_metrics['bbox_mAP_50'],
        mask_metrics['bbox_mAP_75'],
        mask_metrics['bbox_mAP_s'],
        mask_metrics['bbox_mAP_m'],
        mask_metrics['bbox_mAP_l']
    ]
    sparse_scores = [
        sparse_metrics['bbox_mAP'],
        sparse_metrics['bbox_mAP_50'],
        sparse_metrics['bbox_mAP_75'],
        sparse_metrics['bbox_mAP_s'],
        sparse_metrics['bbox_mAP_m'],
        sparse_metrics['bbox_mAP_l']
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))
    bars1 = ax.bar(x - width/2, mask_scores, width, label='Mask R-CNN', color='#89CFF0')  # babyblue
    bars2 = ax.bar(x + width/2, sparse_scores, width, label='Sparse R-CNN', color='#FFB6C1')  # light pink

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    ax.set_ylabel('mAP')
    ax.set_title('COCO-style Evaluation Comparison: Mask R-CNN vs Sparse R-CNN')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 0.65)
    ax.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    print(f"图像已保存到：{save_path}")

    plt.show()
    plt.close()

# 示例调用（你可以替换成你自己的路径）
mask_json = 'work_dirs/mask_rcnn_r50_fpn_1x_voc_lowmem/20250526_110948_test/20250526_110948.json'
sparse_json = 'work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_coco/20250527_105741_test/20250527_105741.json'
save_dir = 'figures'  # 输出文件夹路径
plot_model_comparison(mask_json, sparse_json, save_dir)
