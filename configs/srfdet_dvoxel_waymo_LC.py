plugin = True
plugin_dir = 'mmdet3d_plugin'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.1, 0.1, 0.15]
out_size_factor = 8
point_cloud_range = [-76.8, -76.8, -2, 76.8, 76.8, 4]
sparse_shape = [41, 1536, 1536]
grid_size = [1536, 1536, 40]

lidar_feat_lvls = 4

class_names = ['Car', 'Pedestrian', 'Cyclist']
num_stages = 6

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False)

# Input modality for nuScenes dataset, this is consistent with the submission
# format which requires the information in input_modality.
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='SRFDetWaymo',
    use_img=True,
    freeze_img=False,
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
        voxel_size=voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),
    pts_voxel_encoder=dict(
        type='DynamicVFECustom',
        in_channels=5,
        feat_channels=[5, 5],
        with_distance=False,
        voxel_size=voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1dCustom', eps=1e-3, momentum=0.01),
        # with_centroid_aware_vox=True,
        # centroid_to_point_pos_emb_dims=32,
    ),
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
        # init_cfg=dict(type='Pretrained',
        #               checkpoint='ckpts/futr3d_lidar_cam_new.pth',
        #               prefix='pts_middle_encoder.'),
    ),
    pts_backbone=dict(
        type='SECONDCustom',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        # init_cfg=dict(type='Pretrained',
        #               checkpoint='ckpts/futr3d_lidar_cam_new.pth',
        #               prefix='pts_backbone.'),
    ),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[128, 256],
        out_channels=256,
        start_level=0,
        num_outs=4,
        add_extra_convs='on_output',
    ),
    bbox_head=dict(
        type='SRFDetHead',
        num_classes=len(class_names),
        feat_channels_lidar=256,
        feat_channels_img=256,
        hidden_dim=256,
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
                    embed_dims=128, num_levels=lidar_feat_lvls),
                ffn_cfgs=dict(
                    type='FFN',
                    embed_dims=128,
                    feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                feedforward_channels=256,
                ffn_dropout=0.1,
                operation_order=('self_attn', 'norm', 'ffn', 'norm'))
        ),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        with_dpg=True,
        num_dpg_exp=4,
        single_head_lidar=dict(
            type='SingleSRFDetHead',
            num_cls_convs=2,
            num_reg_convs=3,
            dim_feedforward=1024,
            num_heads=8,
            dropout=0.1,
            bbox_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            use_fusion=True,
        ),
        single_head_img=dict(
            type='SingleSRFDetHeadImg',
            num_cls_convs=2,
            num_reg_convs=3,
            dim_feedforward=1024,
            num_heads=8,
            dropout=0.1,
            act_cfg=dict(type='ReLU', inplace=True),
            dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
            pc_range=point_cloud_range,
        ),
        roi_extractor_lidar=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[8, 16, 32, 64]),
        roi_extractor_img=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        # loss
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='sum',  # remove for Hungarian
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss',
                       reduction='sum',  # remove for Hungarian
                       loss_weight=2.0),
    ),
    # model training and testing settings
    test_cfg=dict(
        use_nms=True,
        use_rotate_nms=True,
        nms_thr=0.4,
        score_thr=0.1,
        min_bbox_size=0,
        max_per_img=300,
        post_center_range=[-80, -80, -10, 80, 80, 10],
    ),
    train_cfg=dict(
        assigner=dict(
            type='OTAssignerSRFDet',
            cls_cost=dict(type='FocalLossCost', alpha=0.25, gamma=2.0,
                          weight=0.6, eps=1e-8),
            reg_cost=dict(type='BBox3DL1Cost', weight=2.0),
            iou_cost=dict(type='IoU3DCost', weight=2.0),
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

dataset_type = 'WaymoDataset'
data_root = 'data/waymo/kitti_format'

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),  # (640, 960)
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
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=6, use_dim=5),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 1333),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ]
    )
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            load_interval=5,
            ann_file=data_root + '/waymo_infos_train.pkl',
            split='training',
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/waymo_infos_val.pkl',
        split='training',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/waymo_infos_val.pkl',
        split='training',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
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
    warmup_iters=20*500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

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

total_epochs = 36
evaluation = dict(interval=21, pipeline=test_pipeline)

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
load_from = './ckpts/msf3detr_voxel_waymo_epoch_36.pth'
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
