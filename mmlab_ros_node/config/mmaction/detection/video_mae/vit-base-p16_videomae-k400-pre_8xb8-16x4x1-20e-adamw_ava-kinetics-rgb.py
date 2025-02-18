# ann_file_train = 'data/ava_kinetics/annotations/ava_train_v2.2.csv'
# ann_file_val = 'data/ava_kinetics/annotations/ava_val_v2.2.csv'
# anno_root = 'data/ava_kinetics/annotations'
# auto_scale_lr = dict(base_batch_size=64, enable=False)
# data_root = 'data/ava_kinetics/rawframes'
# dataset_type = 'AVAKineticsDataset'
# default_hooks = dict(
#     checkpoint=dict(
#         interval=1, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
#     logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     runtime_info=dict(type='RuntimeInfoHook'),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     sync_buffers=dict(type='SyncBuffersHook'),
#     timer=dict(type='IterTimerHook'))
# default_scope = 'mmaction'
# env_cfg = dict(
#     cudnn_benchmark=False,
#     dist_cfg=dict(backend='nccl'),
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
# exclude_file_train = 'data/ava_kinetics/annotations/ava_train_excluded_timestamps_v2.2.csv'
# exclude_file_val = 'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv'
# label_file = 'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt'
# load_from = None
# log_level = 'INFO'
# log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
# model = dict(
#     _scope_='mmdet',
#     backbone=dict(
#         depth=12,
#         drop_path_rate=0.2,
#         embed_dims=768,
#         img_size=224,
#         mlp_ratio=4,
#         norm_cfg=dict(eps=1e-06, type='LN'),
#         num_frames=16,
#         num_heads=12,
#         patch_size=16,
#         qkv_bias=True,
#         return_feat_map=True,
#         type='mmaction.VisionTransformer',
#         use_mean_pooling=False),
#     data_preprocessor=dict(
#         _scope_='mmaction',
#         format_shape='NCTHW',
#         mean=[
#             123.675,
#             116.28,
#             103.53,
#         ],
#         std=[
#             58.395,
#             57.12,
#             57.375,
#         ],
#         type='ActionDataPreprocessor'),
#     init_cfg=dict(
#         checkpoint=
#         'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth',
#         type='Pretrained'),
#     roi_head=dict(
#         bbox_head=dict(
#             dropout_ratio=0.5,
#             in_channels=768,
#             multilabel=True,
#             num_classes=81,
#             type='BBoxHeadAVA'),
#         bbox_roi_extractor=dict(
#             output_size=8,
#             roi_layer_type='RoIAlign',
#             type='SingleRoIExtractor3D',
#             with_temporal_pool=True),
#         type='AVARoIHead'),
#     test_cfg=dict(rcnn=None),
#     train_cfg=dict(
#         rcnn=dict(
#             assigner=dict(
#                 min_pos_iou=0.9,
#                 neg_iou_thr=0.9,
#                 pos_iou_thr=0.9,
#                 type='MaxIoUAssignerAVA'),
#             pos_weight=1.0,
#             sampler=dict(
#                 add_gt_as_proposals=True,
#                 neg_pos_ub=-1,
#                 num=32,
#                 pos_fraction=1,
#                 type='RandomSampler'))),
#     type='FastRCNN')
# optim_wrapper = dict(
#     clip_grad=dict(max_norm=40, norm_type=2),
#     constructor='LearningRateDecayOptimizerConstructor',
#     optimizer=dict(lr=0.000125, type='AdamW', weight_decay=0.05),
#     paramwise_cfg=dict(
#         decay_rate=0.75, decay_type='layer_wise', num_layers=12))
# param_scheduler = [
#     dict(
#         begin=0,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=5,
#         start_factor=0.1,
#         type='LinearLR'),
#     dict(
#         T_max=15,
#         begin=5,
#         by_epoch=True,
#         convert_to_iter_based=True,
#         end=20,
#         eta_min=0,
#         type='CosineAnnealingLR'),
# ]
# proposal_file_train = 'data/ava_kinetics/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl'
# proposal_file_val = 'data/ava_kinetics/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl'
# resume = False
# test_cfg = dict(type='TestLoop')
# test_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
#         data_prefix=dict(img='data/ava_kinetics/rawframes'),
#         exclude_file=
#         'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
#         label_file=
#         'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
#         pipeline=[
#             dict(
#                 clip_len=16,
#                 frame_interval=4,
#                 test_mode=True,
#                 type='SampleAVAFrames'),
#             dict(type='RawFrameDecode'),
#             dict(scale=(
#                 -1,
#                 256,
#             ), type='Resize'),
#             dict(collapse=True, input_format='NCTHW', type='FormatShape'),
#             dict(type='PackActionInputs'),
#         ],
#         proposal_file=
#         'data/ava_kinetics/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
#         test_mode=True,
#         type='AVAKineticsDataset'),
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
# test_evaluator = dict(
#     ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
#     exclude_file=
#     'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
#     label_file=
#     'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
#     type='AVAMetric')
# train_cfg = dict(
#     max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
# train_dataloader = dict(
#     batch_size=8,
#     dataset=dict(
#         ann_file='data/ava_kinetics/annotations/ava_train_v2.2.csv',
#         data_prefix=dict(img='data/ava_kinetics/rawframes'),
#         exclude_file=
#         'data/ava_kinetics/annotations/ava_train_excluded_timestamps_v2.2.csv',
#         label_file=
#         'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
#         pipeline=[
#             dict(clip_len=16, frame_interval=4, type='SampleAVAFrames'),
#             dict(type='RawFrameDecode'),
#             dict(scale_range=(
#                 256,
#                 320,
#             ), type='RandomRescale'),
#             dict(size=256, type='RandomCrop'),
#             dict(flip_ratio=0.5, type='Flip'),
#             dict(collapse=True, input_format='NCTHW', type='FormatShape'),
#             dict(type='PackActionInputs'),
#         ],
#         proposal_file=
#         'data/ava_kinetics/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl',
#         type='AVAKineticsDataset'),
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(shuffle=True, type='DefaultSampler'))
# train_pipeline = [
#     dict(clip_len=16, frame_interval=4, type='SampleAVAFrames'),
#     dict(type='RawFrameDecode'),
#     dict(scale_range=(
#         256,
#         320,
#     ), type='RandomRescale'),
#     dict(size=256, type='RandomCrop'),
#     dict(flip_ratio=0.5, type='Flip'),
#     dict(collapse=True, input_format='NCTHW', type='FormatShape'),
#     dict(type='PackActionInputs'),
# ]
# url = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'
# val_cfg = dict(type='ValLoop')
# val_dataloader = dict(
#     batch_size=1,
#     dataset=dict(
#         ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
#         data_prefix=dict(img='data/ava_kinetics/rawframes'),
#         exclude_file=
#         'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
#         label_file=
#         'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
#         pipeline=[
#             dict(
#                 clip_len=16,
#                 frame_interval=4,
#                 test_mode=True,
#                 type='SampleAVAFrames'),
#             dict(type='RawFrameDecode'),
#             dict(scale=(
#                 -1,
#                 256,
#             ), type='Resize'),
#             dict(collapse=True, input_format='NCTHW', type='FormatShape'),
#             dict(type='PackActionInputs'),
#         ],
#         proposal_file=
#         'data/ava_kinetics/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
#         test_mode=True,
#         type='AVAKineticsDataset'),
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(shuffle=False, type='DefaultSampler'))
# val_evaluator = dict(
#     ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
#     exclude_file=
#     'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
#     label_file=
#     'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
#     type='AVAMetric')
# val_pipeline = [
#     dict(
#         clip_len=16, frame_interval=4, test_mode=True, type='SampleAVAFrames'),
#     dict(type='RawFrameDecode'),
#     dict(scale=(
#         -1,
#         256,
#     ), type='Resize'),
#     dict(collapse=True, input_format='NCTHW', type='FormatShape'),
#     dict(type='PackActionInputs'),
# ]
# vis_backends = [
#     dict(type='LocalVisBackend'),
# ]
# visualizer = dict(
#     type='ActionVisualizer', vis_backends=[
#         dict(type='LocalVisBackend'),
#     ])


default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False


# model settings
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6)),
    cls_head=dict(
        type='TimeSformerHead',
        num_classes=400,
        in_channels=768,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

# dataset settings
dataset_type = 'VideoDataset'
data_root_val = 'data/kinetics400/videos_val'
ann_file_test = 'data/kinetics400/kinetics400_val_list_videos.txt'

test_pipeline = [
    dict(type='DecordInit'),
    dict(
        type='SampleFrames',
        clip_len=16,
        frame_interval=4,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

test_evaluator = dict(type='AccMetric')
test_cfg = dict(type='TestLoop')
url = (
    'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/'
    'vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth')

model = dict(
    type='FastRCNN',
    _scope_='mmdet',
    init_cfg=dict(type='Pretrained', checkpoint=url),
    backbone=dict(
        type='mmaction.VisionTransformer',
        img_size=224,
        patch_size=16,
        embed_dims=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_frames=16,
        norm_cfg=dict(type='LN', eps=1e-6),
        drop_path_rate=0.2,
        use_mean_pooling=False,
        return_feat_map=True),
    roi_head=dict(
        type='AVARoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor3D',
            roi_layer_type='RoIAlign',
            output_size=8,
            with_temporal_pool=True),
        bbox_head=dict(
            type='BBoxHeadAVA',
            background_class=True,
            in_channels=768,
            num_classes=81,
            multilabel=True,
            dropout_ratio=0.5)),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        _scope_='mmaction',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssignerAVA',
                pos_iou_thr=0.9,
                neg_iou_thr=0.9,
                min_pos_iou=0.9),
            sampler=dict(
                type='RandomSampler',
                num=32,
                pos_fraction=1,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=1.0)),
    test_cfg=dict(rcnn=None))

dataset_type = 'AVAKineticsDataset'
data_root = 'data/ava_kinetics/rawframes'
anno_root = 'data/ava_kinetics/annotations'

ann_file_train = f'{anno_root}/ava_train_v2.2.csv'
ann_file_val = f'{anno_root}/ava_val_v2.2.csv'

exclude_file_train = f'{anno_root}/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = f'{anno_root}/ava_val_excluded_timestamps_v2.2.csv'

label_file = f'{anno_root}/ava_action_list_v2.2_for_activitynet_2019.pbtxt'

proposal_file_train = (f'{anno_root}/ava_dense_proposals_train.FAIR.'
                       'recall_93.9.pkl')
proposal_file_val = f'{anno_root}/ava_dense_proposals_val.FAIR.recall_93.9.pkl'

train_pipeline = [
    dict(type='SampleAVAFrames', clip_len=16, frame_interval=4),
    dict(type='RawFrameDecode'),
    dict(type='RandomRescale', scale_range=(256, 320)),
    dict(type='RandomCrop', size=256),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]
# The testing is w/o. any cropping / flipping
val_pipeline = [
    dict(
        type='SampleAVAFrames', clip_len=16, frame_interval=4, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='FormatShape', input_format='NCTHW', collapse=True),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        exclude_file=exclude_file_train,
        pipeline=train_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_train,
        data_prefix=dict(img=data_root)))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        exclude_file=exclude_file_val,
        pipeline=val_pipeline,
        label_file=label_file,
        proposal_file=proposal_file_val,
        data_prefix=dict(img=data_root),
        test_mode=True))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='AVAMetric',
    ann_file=ann_file_val,
    label_file=label_file,
    exclude_file=exclude_file_val)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=20, val_begin=1, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=15,
        eta_min=0,
        by_epoch=True,
        begin=5,
        end=20,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1.25e-4, weight_decay=0.05),
    constructor='LearningRateDecayOptimizerConstructor',
    paramwise_cfg={
        'decay_rate': 0.75,
        'decay_type': 'layer_wise',
        'num_layers': 12
    },
    clip_grad=dict(max_norm=40, norm_type=2))

default_hooks = dict(checkpoint=dict(max_keep_ckpts=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (8 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=64)
