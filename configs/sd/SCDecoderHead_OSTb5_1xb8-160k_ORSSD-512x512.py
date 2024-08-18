norm_cfg = dict(type='SyncBN', requires_grad=True) 
checkpoint = '' # Requires modification
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(512, 512))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        attn_drop_rate=0,
        drop_path_rate=0.1,
        drop_rate=0,
        embed_dims=64,
        in_channels=3,
        init_cfg=dict(
            checkpoint=checkpoint,
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3],
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        patch_sizes=[7, 3, 3, 3],
        qkv_bias=True,
        sr_ratios=[8, 4, 2, 1],
        type='Orientationalstructuretransformer',
        with_cp=True
     ),
    decode_head=dict(
        type='StructureContextualDecoderHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        out_channels=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[dict(type='Guass_Iou_Loss', use_sigmoid=False, loss_weight=1.0, kernel=7, std=10.0)
            ,dict(type='GussL1_Loss', use_sigmoid=False, loss_weight=1.0, kernel=7, std=10.0)]),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'EORSSDDataset'
data_root = ''  #  # Requires modification
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations_255', reduce_zero_label=False),
    dict(
        type='RandomResize',
        scale=crop_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='RandomRotate', prob=0.5, degree=15),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=False),
    dict(type='LoadAnnotations_255', reduce_zero_label=False),
    dict(type='PackSegInputs')
]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=(256, 256), keep_ratio=False),
            dict(type='Resize', scale=(384, 384), keep_ratio=False),
            dict(type='Resize', scale=crop_size, keep_ratio=False),
            dict(type='Resize', scale=(640, 640), keep_ratio=False),
            dict(type='Resize', scale=(768, 768), keep_ratio=False),
            dict(type='Resize', scale=(896, 896), keep_ratio=False),
               ],
            [{
                'type': 'RandomFlip',
                'prob': 0.0,
                'direction': 'horizontal'
            },
                {
                    'type': 'RandomFlip',
                    'prob': 1.0,
                    'direction': 'horizontal'
                },
{
                    'type': 'RandomFlip',
                    'prob': 1.0,
                    'direction': 'vertical'
                }
            ],
            [{
                'type': 'LoadAnnotations_255'
            }], [{
                'type': 'PackSegInputs'
            }]])
]
train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/train/Image-train', seg_map_path='train/train/GT-train'), # Requires modification
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/test/Image-test', seg_map_path='test/test/GT-test'),# Requires modification
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test/test/Image-test', seg_map_path='test/test/GT-test'),# Requires modification
        pipeline=test_pipeline))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore', 'msm'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore', 'msm'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=1.0),
            # pos_block=dict(decay_mult=0.0),
            # norm=dict(decay_mult=0.0),
            head=dict(lr_mult=10.0))),
    # accumulative_counts=2
)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=1e-09,
        power=0.9,
        begin=1500,
        end=160000,
        by_epoch=False)
]
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=200, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, save_best='mSMeasure', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))
launcher = 'none',
randomness = dict(seed=3407)
work_dir = '' # Requires modification
