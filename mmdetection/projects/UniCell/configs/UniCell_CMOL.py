_base_ = [
    './nuclei_det_multihead_cmol.py',
    'mmdet::_base_/default_runtime.py'
]
checkpoint_file = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth'  # noqa
backbone_norm_cfg = dict(type='LN', requires_grad=True)
custom_imports = dict(
    imports=['projects.UniCell.datasets', 'projects.UniCell.deform_module'], allow_failed_imports=False)
num_classes = [3, 4, 2, 6]
weight_classes = [1, 1, 1, 1]

model = dict(
    type='DINO_Nuclei_MultiHead_Feature_4D_CMOL_l_Layers',
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(1, 2, 3),
        window_size=12,
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        use_abs_pos_embed=False,
        convert_weights=True,
        with_cp=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=backbone_norm_cfg,
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    encoder=dict(
        num_layers=3,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4,
                               dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        num_layers=3,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8,
                               dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=4,
                                dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='DINOHead_Nuclei_MultiHead',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        weight_classes=weight_classes,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_point=dict(type='L1Loss', loss_weight=5.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=300)),  # TODO: half num_dn_queries
    text_cfg=dict(
        max_seg_len=77,
        context_length=77,
        width=256,
        layers=3,
        vocab_size=49408,
        proj_num_layers=2,
        n_ctx=16,
        n_query_prompt=16,
        dataset_names = ["CoNSeP", "MoNuSAC", "OCELOT", "Lizard"],
        category_names = ["Lymphoctes", "Epithelial", "Stromal", "Plasma", "Neutrophils", "Eosinophils", "Macrophages",
                          "Tumor"],
        mask_map = {
                    0: [3, 4, 5, 6, 7],
                    1: [2, 3, 5, 7],
                    2: [],
                    3: [6, 7]
                }
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='NucleiL1Cost', weight=5.0),
            ])),
    test_cfg=dict(max_per_img=300))  # 100 for DeformDETR

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scale=(800, 4200),
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(800, 800),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs_Nuclei', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction', 'dataset_id'))
]
train_dataloader = dict(
    batch_size=2,
    # num_workers=0,
    # persistent_workers=False,
    # sampler=dict(_delete_=True, type='ClassAwareSampler_Nuclei'),
    batch_sampler=dict(type='AspectRatioBatchSampler_SameDataset', drop_last=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }))

# learning policy
max_iter = 160000
runner = dict(_delete_=True, type='IterBasedRunner', max_iters=max_iter)
# checkpoint_config = dict(interval=16000)
train_cfg = dict(
    max_iters=max_iter, val_interval=16000, by_epoch=False)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
# evaluation = dict(interval=50, metric='centroids', save_best='F1d', rule='greater')


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=16000, max_keep_ckpts=1, save_best=['Overall_F1d', 'Overall_F1c'],
                    rule='greater', by_epoch=False),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 10,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[80000, 120000],
        gamma=0.1)
]
log_processor = dict(
    by_epoch=False
)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=2)
