ann_file_train = 'data/ava_kinetics/annotations/ava_train_v2.2.csv'
ann_file_val = 'data/ava_kinetics/annotations/ava_val_v2.2.csv'
anno_root = 'data/ava_kinetics/annotations'
auto_scale_lr = dict(base_batch_size=64, enable=False)
data_root = 'data/ava_kinetics/rawframes'
dataset_type = 'AVAKineticsDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=1, max_keep_ckpts=2, save_best='auto', type='CheckpointHook'),
    logger=dict(ignore_last=False, interval=20, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffers=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmaction'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
exclude_file_train = 'data/ava_kinetics/annotations/ava_train_excluded_timestamps_v2.2.csv'
exclude_file_val = 'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv'
label_file = 'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=20)
model = dict(
    _scope_='mmdet',
    backbone=dict(
        depth=12,
        drop_path_rate=0.2,
        embed_dims=768,
        img_size=224,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        num_frames=16,
        num_heads=12,
        patch_size=16,
        qkv_bias=True,
        return_feat_map=True,
        type='mmaction.VisionTransformer',
        use_mean_pooling=False),
    data_preprocessor=dict(
        _scope_='mmaction',
        format_shape='NCTHW',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='ActionDataPreprocessor'),
    init_cfg=dict(
        checkpoint=
        'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth',
        type='Pretrained'),
    roi_head=dict(
        bbox_head=dict(
            dropout_ratio=0.5,
            in_channels=768,
            multilabel=True,
            num_classes=81,
            type='BBoxHeadAVA'),
        bbox_roi_extractor=dict(
            output_size=8,
            roi_layer_type='RoIAlign',
            type='SingleRoIExtractor3D',
            with_temporal_pool=True),
        type='AVARoIHead'),
    test_cfg=dict(rcnn=None),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                min_pos_iou=0.9,
                neg_iou_thr=0.9,
                pos_iou_thr=0.9,
                type='MaxIoUAssignerAVA'),
            pos_weight=1.0,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=32,
                pos_fraction=1,
                type='RandomSampler'))),
    type='FastRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=40, norm_type=2),
    constructor='LearningRateDecayOptimizerConstructor',
    optimizer=dict(lr=0.000125, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        decay_rate=0.75, decay_type='layer_wise', num_layers=12))
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        convert_to_iter_based=True,
        end=5,
        start_factor=0.1,
        type='LinearLR'),
    dict(
        T_max=15,
        begin=5,
        by_epoch=True,
        convert_to_iter_based=True,
        end=20,
        eta_min=0,
        type='CosineAnnealingLR'),
]
proposal_file_train = 'data/ava_kinetics/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl'
proposal_file_val = 'data/ava_kinetics/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
        data_prefix=dict(img='data/ava_kinetics/rawframes'),
        exclude_file=
        'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
        label_file=
        'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                test_mode=True,
                type='SampleAVAFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        'data/ava_kinetics/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        test_mode=True,
        type='AVAKineticsDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
    exclude_file=
    'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
    label_file=
    'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
    type='AVAMetric')
train_cfg = dict(
    max_epochs=20, type='EpochBasedTrainLoop', val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='data/ava_kinetics/annotations/ava_train_v2.2.csv',
        data_prefix=dict(img='data/ava_kinetics/rawframes'),
        exclude_file=
        'data/ava_kinetics/annotations/ava_train_excluded_timestamps_v2.2.csv',
        label_file=
        'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        pipeline=[
            dict(clip_len=16, frame_interval=4, type='SampleAVAFrames'),
            dict(type='RawFrameDecode'),
            dict(scale_range=(
                256,
                320,
            ), type='RandomRescale'),
            dict(size=256, type='RandomCrop'),
            dict(flip_ratio=0.5, type='Flip'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        'data/ava_kinetics/annotations/ava_dense_proposals_train.FAIR.recall_93.9.pkl',
        type='AVAKineticsDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(clip_len=16, frame_interval=4, type='SampleAVAFrames'),
    dict(type='RawFrameDecode'),
    dict(scale_range=(
        256,
        320,
    ), type='RandomRescale'),
    dict(size=256, type='RandomCrop'),
    dict(flip_ratio=0.5, type='Flip'),
    dict(collapse=True, input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
url = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
        data_prefix=dict(img='data/ava_kinetics/rawframes'),
        exclude_file=
        'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
        label_file=
        'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
        pipeline=[
            dict(
                clip_len=16,
                frame_interval=4,
                test_mode=True,
                type='SampleAVAFrames'),
            dict(type='RawFrameDecode'),
            dict(scale=(
                -1,
                256,
            ), type='Resize'),
            dict(collapse=True, input_format='NCTHW', type='FormatShape'),
            dict(type='PackActionInputs'),
        ],
        proposal_file=
        'data/ava_kinetics/annotations/ava_dense_proposals_val.FAIR.recall_93.9.pkl',
        test_mode=True,
        type='AVAKineticsDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='data/ava_kinetics/annotations/ava_val_v2.2.csv',
    exclude_file=
    'data/ava_kinetics/annotations/ava_val_excluded_timestamps_v2.2.csv',
    label_file=
    'data/ava_kinetics/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt',
    type='AVAMetric')
val_pipeline = [
    dict(
        clip_len=16, frame_interval=4, test_mode=True, type='SampleAVAFrames'),
    dict(type='RawFrameDecode'),
    dict(scale=(
        -1,
        256,
    ), type='Resize'),
    dict(collapse=True, input_format='NCTHW', type='FormatShape'),
    dict(type='PackActionInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='ActionVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
