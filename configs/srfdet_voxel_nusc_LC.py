plugin = True
plugin_dir = 'mmdet3d_plugin'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8
# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# sparse_shape = [41, 1440, 1440]
# grid_size = [1440, 1440, 40]
point_cloud_range = [-55.2, -55.2, -5.0, 55.2, 55.2, 3.0]
sparse_shape = [41, 1472, 1472]
grid_size = [1472, 1472, 40]

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)

lidar_feat_lvls = 4
img_feat_lvls = 4

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
num_stages = 6

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='SRFDet',
    use_img=True,
    freeze_img=True,
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=2,
        input_ch=3,
        out_features=['stage2', 'stage3', 'stage4', 'stage5'],
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/dd3d_det_final.pth',
                      prefix='img_backbone.'),
    ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True,
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/dd3d_det_final.pth',
                      prefix='backbone.'),  # backbone for vov
    ),
    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size, max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseEncoderCustom',
        in_channels=5,
        sparse_shape=sparse_shape,
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/futr3d_lidar_cam_new.pth',
                      prefix='pts_middle_encoder.'),
    ),
    pts_backbone=dict(
        type='SECONDCustom',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/futr3d_lidar_cam_new.pth',
                      prefix='pts_backbone.'),
    ),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[128, 256],
        out_channels=128,
        start_level=0,
        num_outs=4,
        add_extra_convs='on_output',
    ),
    bbox_head=dict(
        type='SRFDetHead',
        num_classes=10,
        feat_channels_lidar=128,
        feat_channels_img=256,
        hidden_dim=128,
        lidar_feat_lvls=lidar_feat_lvls,
        img_feat_lvls=4,
        num_proposals=900,
        num_heads=5,
        deep_supervision=True,
        prior_prob=0.01,
        with_lidar_encoder=False,
        grid_size=grid_size,
        out_size_factor=out_size_factor,
        lidar_encoder_cfg=dict(
            type='DetrTransformerEncoder',
            num_layers=2,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiScaleDeformableAttention',
                    embed_dims=256, num_levels=lidar_feat_lvls),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=256,
                    feedforward_channels=512,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=512,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        with_dpg=True,
        num_dpg_exp=4,
        single_head_lidar=dict(
            type='SingleSRFDetHeadLiDAR',
            num_cls_convs=2,
            num_reg_convs=3,
            dim_feedforward=512,
            num_heads=8,
            dropout=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=32, dynamic_num=2),
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
        ),
        single_head_img=dict(
            type='SingleSRFDetHeadImg',
            num_cls_convs=2,
            num_reg_convs=3,
            dim_feedforward=512,
            num_heads=8,
            dropout=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=32, dynamic_num=2),
            pc_range=point_cloud_range,
        ),
        roi_extractor_lidar=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=128,
            featmap_strides=[8, 16, 32, 64]),
        roi_extractor_img=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=128,
            featmap_strides=[4, 8, 16, 32]),
        # loss
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='sum',   # remove for Hungarian
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss',
                       reduction='sum',  # remove for Hungarian
                       loss_weight=0.25),
      ),
    # model training and testing settings
    test_cfg=dict(
        use_nms=True,
        use_rotate_nms=True,
        nms_thr=0.4,
        score_thr=0.1,
        min_bbox_size=0,
        max_per_img=300,
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    ),
    train_cfg=dict(
        assigner=dict(
            type='OTAssignerSRFDet',
            cls_cost=dict(type='FocalLossCost', alpha=0.25, gamma=2.0,
                          weight=2.0, eps=1e-8),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.25),
            center_radius=2.5,
            candidate_topk=8,
            pc_range=point_cloud_range,
            num_heads=6
        ),  # for Hungarian
        # assigner=dict(
        #     type='HungarianAssignerSRFDet',
        #     cls_cost=dict(type='FocalLossCost', weight=0.6),
        #     reg_cost=dict(type='BBox3DL1Cost', weight=2.0))
    )
)

dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d',
                                 'gt_labels_3d'])
]


test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'nuscenes_infos_train.pkl',
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    )
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    # paramwise_cfg=dict(
    #     custom_keys={
    #         'img_backbone': dict(lr_mult=0.1),
    #     }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=10*500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-2)

# optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
# # for 8gpu * 2sample_per_gpu
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 0.0001),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.8947368421052632, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4)

total_epochs = 10
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/SRFDet/srfdet_voxel_nusc_L_fade/epoch_20.pth'
resume_from = None
workflow = [('train', 1)]

find_unused_parameters=True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

freeze_lidar_components = True