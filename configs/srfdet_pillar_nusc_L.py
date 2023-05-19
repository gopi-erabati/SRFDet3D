plugin = True
plugin_dir = 'mmdet3d_plugin'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly

out_size_factor = 2
# point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
# sparse_shape = [41, 1440, 1440]
# grid_size = [1440, 1440, 40]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
grid_size = [512, 512, 40]
voxel_size = [0.2, 0.2, 8]

lidar_feat_lvls = 4

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
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='SRFDet',
    use_img=False,
    freeze_img=False,
    pts_voxel_layer=dict(
        max_num_points=20, voxel_size=voxel_size, max_voxels=(40000, 40000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='PillarFeatureNetCustom',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        point_cloud_range=point_cloud_range,
        legacy=False,
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/pillar_objdgcnn.pth',
                      prefix='pts_voxel_encoder.'),
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=(512, 512)),
    pts_backbone=dict(
        type='SECONDCustom',
        in_channels=64,
        out_channels=[64, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False),
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/pillar_objdgcnn.pth',
                      prefix='pts_backbone.'),
    ),
    pts_neck=dict(
        type='FPN',
        norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU'),
        in_channels=[64, 128, 256],
        out_channels=128,
        start_level=0,
        num_outs=4,
        init_cfg=dict(type='Pretrained',
                      checkpoint='ckpts/pillar_objdgcnn.pth',
                      prefix='pts_neck.'),
    ),
    bbox_head=dict(
        type='SRFDetHead',
        num_classes=10,
        feat_channels_lidar=128,
        feat_channels_img=256,
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
            out_channels=128,
            featmap_strides=[2, 4, 8, 16]),
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
        #     cls_cost=dict(type='FocalLossCost', weight=2.0),
        #     reg_cost=dict(type='BBox3DL1Cost', weight=0.25))
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
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    # dict(
    #     type='ObjectSample',
    #     db_sampler=dict(
    #         data_root=data_root,
    #         info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    #         rate=1.0,
    #         prepare=dict(
    #             filter_by_difficulty=[-1],
    #             filter_by_min_points=dict(
    #                 car=5,
    #                 truck=5,
    #                 bus=5,
    #                 trailer=5,
    #                 construction_vehicle=5,
    #                 traffic_cone=5,
    #                 barrier=5,
    #                 motorcycle=5,
    #                 bicycle=5,
    #                 pedestrian=5)),
    #         classes=class_names,
    #         sample_groups=dict(
    #             car=2,
    #             truck=3,
    #             construction_vehicle=7,
    #             bus=4,
    #             trailer=6,
    #             barrier=2,
    #             motorcycle=6,
    #             bicycle=6,
    #             pedestrian=2,
    #             traffic_cone=2),
    #         points_loader=dict(
    #             type='LoadPointsFromFile',
    #             coord_type='LIDAR',
    #             load_dim=5,
    #             use_dim=[0, 1, 2, 3, 4],
    #             file_client_args=file_client_args
    #         ))),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
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
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=6,
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
            box_type_3d='LiDAR'),
    ),
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
    warmup_iters=4*500,
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

total_epochs = 20
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
load_from = None
resume_from = None
workflow = [('train', 1)]

find_unused_parameters = True
# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
