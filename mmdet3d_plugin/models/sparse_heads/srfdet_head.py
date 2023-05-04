import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, BaseModule, ModuleList
from mmcv.cnn import build_activation_layer, ConvModule, build_conv_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.ops import MultiScaleDeformableAttention

from mmdet.core import build_assigner, bbox2roi, multi_apply, build_sampler
from mmdet.core.utils import reduce_mean
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.models import HEADS, build_loss, build_head, build_roi_extractor

from ...core.bbox.util import (normalize_bbox, denormalize_bbox,
                               boxes3d_to_corners3d)

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward
        Args:
            xyz (tensor): shape (BS, n_q, 2)
        """
        xyz = xyz.transpose(1, 2).contiguous()  # (BS, 2, n_q)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding  # (BS, 128, n_q)


@HEADS.register_module()
class SRFDetHead(BaseDenseHead):
    """
    This is the head for SRFDet
    """

    def __init__(self,
                 use_img=False,
                 num_classes=4,
                 feat_channels_lidar=256,
                 feat_channels_img=256,
                 hidden_dim=128,
                 lidar_feat_lvls=4,
                 img_feat_lvls=4,
                 num_proposals=128,
                 num_heads=6,
                 deep_supervision=True,
                 prior_prob=0.01,
                 with_lidar_encoder=False,
                 grid_size=None,
                 out_size_factor=8,
                 lidar_encoder_cfg=None,
                 code_weights=None,
                 with_dpg=True,
                 num_dpg_exp=4,
                 single_head_lidar=None,
                 single_head_img=None,
                 roi_extractor_lidar=None,
                 roi_extractor_img=None,
                 # loss
                 sync_cls_avg_factor=True,
                 loss_cls=None,
                 loss_bbox=None,
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None, ):
        super(SRFDetHead, self).__init__(init_cfg)

        self.num_classes = num_classes
        self.use_img = use_img
        self.feat_channels_lidar = feat_channels_lidar
        self.feat_channels_img = feat_channels_img
        self.hidden_dim = hidden_dim
        self.lidar_feat_lvls = lidar_feat_lvls
        self.img_feat_lvls = img_feat_lvls
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.deep_supervision = deep_supervision
        self.prior_prob = prior_prob
        self.code_weights = code_weights
        self.pc_range = single_head_lidar['pc_range']
        self.test_cfg = test_cfg
        self.with_dpg = with_dpg  # Dynamic Proposal Generation
        self.num_dpg_exp = num_dpg_exp
        self.grid_size = grid_size
        self.out_size_factor = out_size_factor
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.with_lidar_encoder = with_lidar_encoder
        self.lidar_encoder_cfg = lidar_encoder_cfg
        # Build Encoder for LiDAR BEV
        if self.with_lidar_encoder:
            self._build_lidar_encoder()

        # Dynamic Proposal Generation
        if self.with_dpg:
            self._build_dynamic_prop_gen()
        else:
            self.init_proposal_boxes = nn.Embedding(self.num_proposals,
                                                    len(self.code_weights))
            self.init_proposal_feats = nn.Embedding(self.num_proposals,
                                                    self.feat_channels_lidar)

        self.use_fed_loss = False
        self.use_focal_loss = True

        # Build Dynamic Head LiDAR
        single_head_lidar_ = single_head_lidar.copy()
        single_head_lidar_.update(num_classes=num_classes)
        single_head_lidar_.update(feat_channels=feat_channels_lidar)
        default_pooler_resolution = roi_extractor_lidar['roi_layer'].get(
            'output_size')
        single_head_lidar_.update(pooler_resolution=default_pooler_resolution)
        single_head_lidar_.update(
            use_focal_loss=self.use_focal_loss, use_fed_loss=self.use_fed_loss)
        single_head_module_lidar = build_head(single_head_lidar_)
        self.head_series_lidar = ModuleList(
            [copy.deepcopy(single_head_module_lidar) for _ in
             range(num_heads)])
        # Build ROI Extractor
        self.roi_extractor_lidar = build_roi_extractor(roi_extractor_lidar)

        if self.use_img:
            # conv to reduce channels of feat map of img
            self.img_convs = nn.ModuleList()
            for _ in range(img_feat_lvls):
                self.img_convs.append(build_conv_layer(
                    dict(type='Conv2d'),
                    feat_channels_img,  # channel of img feature map
                    hidden_dim,  # 128
                    kernel_size=3,
                    padding=1,
                    bias='auto',
                ))

            # Build Dynamic Head Image
            single_head_img_ = single_head_img.copy()
            single_head_img_.update(num_classes=num_classes)
            single_head_img_.update(feat_channels=hidden_dim)
            default_pooler_resolution = roi_extractor_img['roi_layer'].get(
                'output_size')
            single_head_img_.update(
                pooler_resolution=default_pooler_resolution)
            single_head_img_.update(
                use_focal_loss=self.use_focal_loss,
                use_fed_loss=self.use_fed_loss)
            single_head_module_img = build_head(single_head_img_)
            self.head_series_img = ModuleList(
                [copy.deepcopy(single_head_module_img) for _ in
                 range(num_heads)])
            # ROI Extractor
            self.roi_extractor_img = build_roi_extractor(roi_extractor_img)

        # Build Losses
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        # Build Assigner
        if train_cfg:
            self.assigner_type = train_cfg.assigner.type
            self.assigner = build_assigner(train_cfg.assigner)

            # for Hungarian
            if self.assigner_type == 'HungarianAssignerSRFDet':
                # sampler
                sampler_cfg = dict(type='PseudoSampler')
                self.sampler = build_sampler(sampler_cfg, context=self)
                # Background Class weights
                self.bg_cls_weight = 0
                # classes
                if loss_cls:
                    if loss_cls.use_sigmoid:
                        self.cls_out_channels = num_classes
                    else:
                        self.cls_out_channels = num_classes + 1

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        # NMS
        self.use_nms = self.test_cfg.get('use_nms', True)
        self._init_weights()  # No for srcn3d

    def _init_weights(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for name, p in self.named_parameters():
            if name in ['code_weights', 'init_proposal_boxes.weight',
                        'init_proposal_feats.weight']:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal_loss or self.use_fed_loss:
                if (p.shape[-1] == self.num_classes or \
                    p.shape[-1] == self.num_classes + 1) and (
                        not name == 'code_weights'):
                    nn.init.constant_(p, bias_value)
        if self.with_lidar_encoder:
            nn.init.normal_(self.bev_level_embeds)
            for m in self.modules():
                if isinstance(m, MultiScaleDeformableAttention):
                    m.init_weights()

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base  # (1, x_size * y_size, 2)

    def _build_lidar_encoder(self):
        self.encoder_lidar = build_transformer_layer_sequence(
            self.lidar_encoder_cfg)

        # BEV position Embeddings
        self.bev_pos_encoder_mlvl_embed = ModuleList()
        for _ in range(self.lidar_feat_lvls):
            self.bev_pos_encoder_mlvl_embed.append(
                PositionEmbeddingLearned(
                    2, self.feat_channels_lidar))

        # BEV Level Embeddings
        self.bev_level_embeds = nn.Parameter(torch.Tensor(
            self.lidar_feat_lvls, self.feat_channels_lidar))

        # BEV Pos for Multi-levels
        x_size = self.grid_size[0] // self.out_size_factor
        y_size = self.grid_size[1] // self.out_size_factor
        self.bev_pos_mlvl = []
        for lvl in range(self.lidar_feat_lvls):
            self.bev_pos_mlvl.append(
                self.create_2D_grid(int(x_size / (2 ** lvl)),
                                    int(y_size / (2 ** lvl))))

    def _build_dynamic_prop_gen(self):
        self.init_proposal_boxes = nn.Embedding(self.num_dpg_exp *
                                                self.num_proposals,
                                                len(self.code_weights))
        self.init_proposal_feats = nn.Embedding(self.num_dpg_exp *
                                                self.num_proposals,
                                                self.feat_channels_lidar)

        # build DPG Staircase layers
        self.dpg_dw_convs_lidar = ModuleList()
        for lvl in range(self.lidar_feat_lvls - 1):
            dw_conv_lidar = ConvModule(
                in_channels=self.feat_channels_lidar * (lvl + 1),
                out_channels=self.feat_channels_lidar * (lvl + 1),
                kernel_size=3,
                stride=2,
                padding=1,
                groups=self.feat_channels_lidar * (lvl + 1),
                norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
            )
            self.dpg_dw_convs_lidar.append(dw_conv_lidar)
        last_fmap_size = int(self.grid_size[0] / (self.out_size_factor * (2
                                                                          ** (self.lidar_feat_lvls - 1))))
        self.dpg_fc1_lidar = nn.Linear(last_fmap_size*last_fmap_size, 1024)
        self.dpg_act_lidar = nn.ReLU(inplace=True)
        self.dpg_fc2_lidar = nn.Linear(1024,
                                       self.num_dpg_exp * self.num_proposals)

        # if also with image
        if self.use_img:
            # build DPG Staircase layers
            self.dpg_dw_convs_img = ModuleList()
            for lvl in range(self.img_feat_lvls - 1):
                dw_conv_img = ConvModule(
                    in_channels=self.hidden_dim * (lvl + 1),
                    out_channels=self.hidden_dim * (lvl + 1),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=self.hidden_dim * (lvl + 1),
                    norm_cfg=dict(type='BN2d', eps=1e-3, momentum=0.01),
                )
                self.dpg_dw_convs_img.append(dw_conv_img)
            self.dpg_fc1_img = nn.Linear(900, 1500)
            self.dpg_act_img = nn.ReLU(inplace=True)
            self.dpg_fc2_img = nn.Linear(1500,
                                         self.num_dpg_exp * self.num_proposals)

    def forward_train(self,
                      img_feats, point_feats,
                      gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None,
                      img_metas=None,
                      proposal_cfg=None,
                      **kwargs):
        """Forward function for training mode.

        Args:
            img_feats (list[Tensor] | None): Image feats list of stride 4, 8,
                16, 32 of shape (B * N, 128, H, W)
            point_feats (list[Tensor]): Point feat list [(B, 128, H, W)...]
                strides 8, 16, 32, 64 of 1472
            gt_bboxes (list[Tensor]): Ground truth bboxes ,
                shape (num_gts, 7).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 7).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert proposal_cfg is None, '"proposal_cfg" must be None'

        pred_logits, pred_bboxes = self(img_feats, point_feats, img_metas)
        # (#lay, bs, n_p, #cls), (#lay, bs, n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }
        if self.deep_supervision:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]
        # output = {'pred_logits':(bs, n_p, #cls),
        #           'pred_boxes':(bs, n_p, 10),
        #           'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
        #                            'pred_boxes':(bs, n_p, 10)},
        #                            { }, { }, ...]}
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        if self.assigner_type == 'HungarianAssignerSRFDet':
            losses = self.loss_hung(output, gt_bboxes, gt_labels)
        elif self.assigner_type == 'OTAssignerSRFDet':
            losses = self.loss_ota(output, gt_bboxes, gt_labels)

        return losses

    def forward(self, img_feats, point_feats, img_metas):
        """
        Forward Function
        Args:
            img_feats (list[(Tensor)]): Features of imgs from backbone
                (B*N, C, H, W), strides 4, 8, 16, 32
            point_feats (list[Tensor]): BEV LiDAR feats of shape
                [(B, 128, H, W)...] strides 8, 16, 32, 64 of 1472
            img_metas (list): list of img_meta for each image

        Returns:
            pred_logits (Tensor): predictions (logit) of shape
                (#lay, bs, n_p, #cls)
            pred_boxes (Tensor): box predictions of shape (#lay, bs, n_p, 10)
        """

        inter_pred_logits = []
        inter_pred_bboxes = []

        # Encoder for LiDAR BEV feats if with_lidar_encoder=True
        if self.with_lidar_encoder:
            point_feats = self._get_lidar_encoder_feats(point_feats)
            # (list[Tensor]): shape (bs, C, H, W)

        # convert image feat dim to lidar feat dim (128)
        if self.use_img:
            # convert img_feats chnls to hidden_chnls
            for feat_idx, img_feat in enumerate(img_feats):
                bs, n_cam, C, H, W = img_feat.shape
                img_feat = img_feat.reshape(bs * n_cam, C, H, W)
                img_feats[feat_idx] = self.img_convs[feat_idx](img_feat)
                BN, C, H, W = img_feats[feat_idx].shape
                img_feats[feat_idx] = img_feats[feat_idx].reshape(bs, int(BN /
                                                                          bs),
                                                                  C,
                                                                  H, W)

        # Get init proposals
        init_proposal_boxes, proposal_feats = self._get_init_proposals(
            img_feats, point_feats)
        # (bs, n_p, 10), (bs, n_p, dim)
        bboxes = init_proposal_boxes

        # apply sigmoid for box center
        bboxes[..., :3] = bboxes[..., :3].sigmoid()

        # Head for LiDAR
        for head_idx_lidar, single_head_lidar in enumerate(
                self.head_series_lidar):
            pred_logits, pred_bboxes, proposal_feats = single_head_lidar(
                point_feats, bboxes, proposal_feats, self.roi_extractor_lidar,
                img_metas)
            # (bs, n_p, #cls), (bs, n_p, 10), (bs*n_p, 256)
            # pred boxes  center:norm and size:log
            # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            if self.deep_supervision and not self.use_img:
                inter_pred_logits.append(pred_logits)
                inter_pred_bboxes.append(pred_bboxes.clone())
            bboxes = pred_bboxes.clone().detach()

        # if with image
        if self.use_img:
            for head_idx_img, single_head_img in enumerate(
                    self.head_series_img):
                pred_logits, pred_bboxes, proposal_feats = single_head_img(
                    img_feats, bboxes, proposal_feats, self.roi_extractor_img,
                    img_metas)
                # (bs, n_p, #cls), (bs, n_p, 10), (bs*n_p, 256)
                # pred boxes  center:norm and size:log
                # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
                if self.deep_supervision:
                    inter_pred_logits.append(pred_logits)
                    inter_pred_bboxes.append(pred_bboxes.clone())
                bboxes = pred_bboxes.clone().detach()

        if self.deep_supervision:
            inter_pred_logits = torch.stack(inter_pred_logits)
            inter_pred_bboxes = torch.stack(inter_pred_bboxes)
            # center denormalize
            pc_range_ = point_feats[0].new_tensor(
                [[self.pc_range[3] - self.pc_range[0],
                  self.pc_range[4] - self.pc_range[1],
                  self.pc_range[5] - self.pc_range[2]]])  # (1, 3)
            pc_start_ = point_feats[0].new_tensor(
                [[self.pc_range[0], self.pc_range[1], self.pc_range[2]]])
            # (1, 3)
            inter_pred_bboxes[..., :3] = (inter_pred_bboxes[...,
                                          :3] * pc_range_) + \
                                         pc_start_
            return inter_pred_logits, inter_pred_bboxes
            # (#lay, bs, n_p, #cls), (#lay, bs, n_p, 10)
            # pred boxes  center:abs and size:log
            # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        else:
            # center denormalize
            pc_range_ = point_feats[0].new_tensor(
                [[self.pc_range[3] - self.pc_range[0],
                  self.pc_range[4] - self.pc_range[1],
                  self.pc_range[5] - self.pc_range[2]]])  # (1, 3)
            pc_start_ = point_feats[0].new_tensor(
                [[self.pc_range[0], self.pc_range[1], self.pc_range[2]]])
            # (1, 3)
            pred_bboxes[..., :3] = (pred_bboxes[..., :3] * pc_range_) + \
                                   pc_start_
            return pred_logits[None, ...], pred_bboxes[None, ...]

    def _get_init_proposals(self, img_feats, point_feats):
        """ Function to get Dynamic Proposal generation proposals
        or
        General proposals

        Args:
            img_feats (list[Tensor]): of shape (bs, C, H, W)
            point_feats (list[Tensor]): of shape (bs, C, H, W)

        Returns:
            (tuple):
                init_proposal_boxes (Tensor): of shape (bs, n_p, 10)
                init_proposal_feats (Tensor): of shape (bs, n_p, dim)
        """
        batch_size = point_feats[0].shape[0]

        # Dynamic Proposal Generation (DPG)
        if self.with_dpg:
            # initial H, W of feat map is 8x stride of grid size which is 184
            pfeat_1_lidar = self.dpg_dw_convs_lidar[0](point_feats[0])
            # (bs, C, H/2, W/2)
            pfeat_12_lidar = torch.cat([point_feats[1], pfeat_1_lidar], dim=1)
            # (bs, 2C, H/2, W/2)
            pfeat_2_lidar = self.dpg_dw_convs_lidar[1](pfeat_12_lidar)
            # (bs, 2C, H/4, W/4)
            pfeat_23_lidar = torch.cat([point_feats[2], pfeat_2_lidar], dim=1)
            # (bs, 3C, H/4, W/4)
            pfeat_3_lidar = self.dpg_dw_convs_lidar[2](pfeat_23_lidar)
            # (bs, 3C, H/8, W/8)
            pfeat_34_lidar = torch.cat([point_feats[3], pfeat_3_lidar], dim=1)
            # (bs, 4C, H/8, W/8)
            dpg_weights_lidar = pfeat_34_lidar.sum(
                dim=1)  # (bs, H/8, W/8) - 23x23
            dpg_weights_lidar = dpg_weights_lidar.flatten(1,
                                                          2)  # (bs, H*W) - 23x23
            dpg_weights_lidar = self.dpg_fc1_lidar(
                dpg_weights_lidar)  # (bs, 1024)
            dpg_weights_lidar = self.dpg_act_lidar(dpg_weights_lidar)
            dpg_weights_lidar = self.dpg_fc2_lidar(
                dpg_weights_lidar)  # (bs, n_dpg_exp * n_p)
            dpg_weights_lidar = dpg_weights_lidar.reshape(batch_size,
                                                          self.num_dpg_exp,
                                                          self.num_proposals)
            # (bs, n_dpg_exp, n_p)

            # image dpg
            if self.use_img:
                # initial H, W of feat map is 8x stride of grid size which is 184
                B, N, C, H, W = img_feats[0].size()
                img_feats_0 = img_feats[0].view(B * N, C, H, W)
                B, N, C, H, W = img_feats[1].size()
                img_feats_1 = img_feats[1].view(B * N, C, H, W)
                B, N, C, H, W = img_feats[2].size()
                img_feats_2 = img_feats[2].view(B * N, C, H, W)
                B, N, C, H, W = img_feats[3].size()
                img_feats_3 = img_feats[3].view(B * N, C, H, W)

                pfeat_1_img = self.dpg_dw_convs_img[0](img_feats_0)
                # (bs, C, H/2, W/2)
                pfeat_12_img = torch.cat([img_feats_1, pfeat_1_img],
                                         dim=1)
                # (bs, 2C, H/2, W/2)
                pfeat_2_img = self.dpg_dw_convs_img[1](pfeat_12_img)
                # (bs, 2C, H/4, W/4)
                pfeat_23_img = torch.cat([img_feats_2, pfeat_2_img],
                                         dim=1)
                # (bs, 3C, H/4, W/4)
                pfeat_3_img = self.dpg_dw_convs_img[2](pfeat_23_img)
                # (bs, 3C, H/8, W/8)
                pfeat_34_img = torch.cat([img_feats_3, pfeat_3_img],
                                         dim=1)
                # (bs, 4C, H/8, W/8)
                # interpolate to 30 x 30
                pfeat_34_img = F.interpolate(pfeat_34_img, [30, 30])
                # (bs*N, 4C, 30, 30)
                BN, C, H, W = pfeat_34_img.shape
                pfeat_34_img = pfeat_34_img.view(B, int(BN / B), C, H, W)
                pfeat_34_img = pfeat_34_img.sum(dim=1)  # (bs, C, H, W)
                dpg_weights_img = pfeat_34_img.sum(
                    dim=1)  # (bs, 30, 30)
                dpg_weights_img = dpg_weights_img.flatten(1, 2)  # (bs, 30*30)
                dpg_weights_img = self.dpg_fc1_img(
                    dpg_weights_img)  # (bs, 1500)
                dpg_weights_img = self.dpg_act_img(dpg_weights_img)
                dpg_weights_img = self.dpg_fc2_img(
                    dpg_weights_img)  # (bs, n_dpg_exp * n_p)
                dpg_weights_img = dpg_weights_img.reshape(batch_size,
                                                          self.num_dpg_exp,
                                                          self.num_proposals)
                # (bs, n_dpg_exp, n_p)
                dpg_weights = (dpg_weights_lidar + dpg_weights_img) / 2
            else:
                dpg_weights = dpg_weights_lidar

            dpg_weights = dpg_weights.softmax(1)  # (bs, n_dpg_exp, n_p)

            # get the proposal embedding weight
            init_proposal_boxes = self.init_proposal_boxes.weight
            # (n_dpg_exp * n_p, 10)
            init_proposal_boxes = init_proposal_boxes.view(self.num_dpg_exp,
                                                           self.num_proposals,
                                                           len(self.code_weights))
            # (n_dpg_exp, n_p, 10)
            init_proposal_feats = self.init_proposal_feats.weight
            # (n_dpg_exp * n_p, dim)
            init_proposal_feats = init_proposal_feats.view(self.num_dpg_exp,
                                                           self.num_proposals,
                                                           self.feat_channels_lidar)
            # (n_dpg_exp, n_p, dim)

            # repeat for all batches
            init_proposal_boxes = init_proposal_boxes.repeat(batch_size,
                                                             1, 1, 1).view(
                batch_size, init_proposal_boxes.shape[0],
                init_proposal_boxes.shape[1],
                init_proposal_boxes.shape[2])  # (bs, n_dpg_exp, n_p, 10)
            init_proposal_feats = init_proposal_feats.repeat(batch_size,
                                                             1, 1, 1).view(
                batch_size, init_proposal_feats.shape[0],
                init_proposal_feats.shape[1],
                init_proposal_feats.shape[2])  # (bs, n_dpg_exp, n_p, dim)

            # multiply the weights with proposals and add them
            init_proposal_boxes = dpg_weights.unsqueeze(
                -1) * init_proposal_boxes
            # (bs, n_dpg_exp, n_p, 10)
            init_proposal_feats = dpg_weights.unsqueeze(
                -1) * init_proposal_feats
            # (bs, n_dpg_exp, n_p, dim)
            init_proposal_boxes = init_proposal_boxes.sum(1)  # (bs, n_p, 10)
            init_proposal_feats = init_proposal_feats.sum(1)  # (bs, n_p, dim)

        else:
            init_proposal_boxes = self.init_proposal_boxes.weight
            init_proposal_boxes = init_proposal_boxes.repeat(batch_size,
                                                             1).view(
                batch_size, init_proposal_boxes.shape[0],
                init_proposal_boxes.shape[1])  # (bs, n_p, 10)
            init_proposal_feats = self.init_proposal_feats.weight
            init_proposal_feats = init_proposal_feats.repeat(batch_size,
                                                             1).view(
                batch_size, init_proposal_feats.shape[0],
                init_proposal_feats.shape[1])  # (bs, n_p, dim)

        return init_proposal_boxes, init_proposal_feats
        # (bs, n_p, 10),(bs, n_p, dim)

    def _get_lidar_encoder_feats(self, lidar_feats):
        """
        This function is to get LiDAR BEV Encoder
        features with MultiScaleDeformAttn
        """
        batch_size = lidar_feats[0].shape[0]
        # repeat the BEV positions for all batches
        bev_pos_mlvl_bs = []
        for bev_pos_lvl in self.bev_pos_mlvl:
            bev_pos_lvl = bev_pos_lvl.repeat(batch_size, 1, 1).to(
                lidar_feats[0].device)  # (bs, H*W, 2)
            bev_pos_mlvl_bs.append(bev_pos_lvl)

        # Encoder: MS Deformable Attention for LiDAR features
        # get the BEV positions and embeddings of all levels with level
        # embed and also feats of all levels flatten
        bev_pos_encoder_mlvl_norm = []
        bev_pos_encoder_mlvl_embed = []
        bev_spatial_shape_mlvl = []
        lidar_feat_mlvl = []
        for idx, (bev_pos_lvl, lidar_feat) in enumerate(
                zip(bev_pos_mlvl_bs, lidar_feats)):
            bev_pos_encoder_lvl_embed = self.bev_pos_encoder_mlvl_embed[
                idx](bev_pos_lvl)  # (bs, h_dim, H*W)
            bev_pos_encoder_lvl_embed = \
                bev_pos_encoder_lvl_embed.permute(0, 2, 1)
            # (bs, H*W, h_dim)
            bev_pos_encoder_lvl_embed = bev_pos_encoder_lvl_embed + \
                                        self.bev_level_embeds[idx].view(
                                            1, 1, -1)  # (bs, H*W, h_dim)
            bev_pos_encoder_mlvl_embed.append(bev_pos_encoder_lvl_embed)

            # LiDAR feats
            lidar_feat_bs, lidar_feat_dim, lidar_feat_h, lidar_feat_w = \
                lidar_feat.shape
            bev_spatial_shape = (lidar_feat_h, lidar_feat_w)
            bev_spatial_shape_mlvl.append(bev_spatial_shape)
            lidar_feat = lidar_feat.flatten(2).permute(0, 2, 1)
            # (bs, H*W, h_dim)
            lidar_feat_mlvl.append(lidar_feat)

            # normalize bev_pos_encoder_lvl with lidar_feat_h and
            # lidar_feat_w to make them lie in [0, 1] for reference points
            bev_pos_encoder_lvl_norm = bev_pos_lvl.float()
            bev_pos_encoder_lvl_norm[..., 0] /= lidar_feat_h
            bev_pos_encoder_lvl_norm[..., 1] /= lidar_feat_w
            bev_pos_encoder_mlvl_norm.append(bev_pos_encoder_lvl_norm)

        # concatenate all levels
        lidar_feat_mlvl = torch.cat(lidar_feat_mlvl, dim=1)
        # (bs, lvl*H*W, h_dim)
        bev_pos_encoder_mlvl_norm = torch.cat(bev_pos_encoder_mlvl_norm,
                                              dim=1)
        # (bs, lvl*H*W, 2) normalized
        # repeat the bev_pos_encoder_mlvl (reference points) for all levels
        bev_pos_encoder_mlvl_norm = \
            bev_pos_encoder_mlvl_norm.unsqueeze(2).repeat(1, 1,
                                                          len(lidar_feats),
                                                          1)
        # (bs, lvl*H*W, lvls, 2)  normalized for reference points
        bev_pos_encoder_mlvl_embed = torch.cat(
            bev_pos_encoder_mlvl_embed, dim=1)  # (bs, lvl*H*W, h_dim)
        bev_spatial_shape_mlvl_tensor = torch.as_tensor(
            bev_spatial_shape_mlvl, dtype=torch.long,
            device=lidar_feat_mlvl.device)  # (lvl, 2)
        bev_level_start_index = torch.cat(
            (bev_spatial_shape_mlvl_tensor.new_zeros(
                (1,)),
             bev_spatial_shape_mlvl_tensor.prod(1).cumsum(0)[
             :-1]))  # (lvl, )

        # reshape according to encoder expectation
        lidar_feat_mlvl = lidar_feat_mlvl.permute(1, 0, 2)
        # (lvl*H*W, bs, h_dim)
        bev_pos_encoder_mlvl_embed = bev_pos_encoder_mlvl_embed.permute(
            1, 0, 2)
        # (lvl*H*W, bs, h_dim)
        lidar_feat_mlvl_encoder = self.encoder_lidar(
            query=lidar_feat_mlvl,
            key=None,
            value=None,
            query_pos=bev_pos_encoder_mlvl_embed,
            spatial_shapes=bev_spatial_shape_mlvl_tensor,
            reference_points=bev_pos_encoder_mlvl_norm,
            level_start_index=bev_level_start_index
        )
        # (lvl*H*W, bs, h_dim)

        # bring back the shape of feature maps
        lidar_feat_mlvl_encoder_list = lidar_feat_mlvl_encoder.split(
            [H_ * W_ for H_, W_ in bev_spatial_shape_mlvl],
            dim=0)
        # [(H*W, bs, h_dim), (H*W, bs, h_dim), ...]
        lidar_feats = []
        for level, (H_, W_) in enumerate(bev_spatial_shape_mlvl):
            memory_point_fmap = lidar_feat_mlvl_encoder_list[
                level].permute(
                1, 2, 0).reshape(lidar_feat_bs, lidar_feat_dim, H_, W_)
            lidar_feats.append(memory_point_fmap)
            # this contains list [(bs, c, h, w), ... for levels]

        return lidar_feats

    @force_fp32(apply_to='outputs')
    def loss_hung(self, outputs, gt_bboxes_list, gt_labels_list):
        """
        This is the loss for DDet3D

        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes absolute [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            gt_bboxes_list (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            gt_labels_list (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )

        Returns:
            (dict) A dict of loss with 'loss_cls', 'loss_bbox', 'd0.loss_cls',
            'd0.loss_bbox', ...
        """
        # convert outputs to the format this loss expects
        all_sem_cls_logits = [outputs['pred_logits']]
        all_bbox_pred = [outputs['pred_boxes']]
        for aux_output in outputs['aux_outputs']:
            all_sem_cls_logits.append(aux_output['pred_logits'])
            all_bbox_pred.append(aux_output['pred_boxes'])
        all_sem_cls_logits = torch.stack(all_sem_cls_logits)
        # (#lay, bs, n_p, #cls)
        all_bbox_pred = torch.stack(all_bbox_pred)  # (#lay, bs, n_p, 10)

        num_dec_layers = len(all_sem_cls_logits)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_sem_cls_logits,
            all_bbox_pred,
            all_gt_bboxes_list,
            all_gt_labels_list)

        loss_dict = dict()

        # loss from last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for (loss_cls_i, loss_bbox_i) in zip(
                losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    sem_cls_logits,
                    bbox_pred,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """Loss function for each decoder layer
            Args:
                sem_cls_logits (Tensor): class logits (B, n_q, #cls)
                bbox_pred (Tensor): Bboxes  (B, n_q, 10)
                gt_bboxes_list (list[Tensor]): Ground truth bboxes for each
                    image with shape (num_gts, 7) in [cx, cy, cz, l, w, h,
                    theta] format. LiDARInstance3DBoxes
                gt_labels_list (list[Tensor]): Ground truth class indices
                    for each image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                (tuple): loss_cls, loss_bbox
        """

        num_imgs = sem_cls_logits.size(0)

        # prepare scores and bboxes list for all images to get targets
        sem_cls_logits_list = [sem_cls_logits[i] for i in range(num_imgs)]
        # [(n_q, #cls+1), ... #images]
        bbox_pred_list = [bbox_pred[i] for i in range(num_imgs)]
        # [(n_q, 10), ... #images]

        cls_reg_targets = self.get_targets(sem_cls_logits_list,
                                           bbox_pred_list,
                                           gt_bboxes_list,
                                           gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)  # (bs*num_q, )
        label_weights = torch.cat(label_weights_list, 0)  # (bs*num_q, )
        bbox_targets = torch.cat(bbox_targets_list, 0)  # (bs * num_q, 8)
        bbox_weights = torch.cat(bbox_weights_list, 0)  # (bs * num_q, 8)

        # classification loss
        pred_logits = sem_cls_logits.reshape(-1, self.cls_out_channels)
        # (bs * num_q, #cls)
        # construct weighted avg_factor
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                pred_logits.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            pred_logits, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_pred.reshape(-1, bbox_pred.size(-1))
        # (bs * num_q, 10)
        normalized_bbox_targets = normalize_bbox(bbox_targets, None)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :len(self.code_weights)],
            normalized_bbox_targets[isnotnan, :len(self.code_weights)],
            bbox_weights[isnotnan, :len(self.code_weights)],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    def get_targets(self,
                    sem_cls_logits_list,
                    bbox_pred_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """
        Compute Classification and Regression targets for all batch elements.

        Args:
            sem_cls_logits_list (list[Tensor]): Box score logits for each
                batch element (n_q, #cls)
            bbox_pred_list (list[Tensor]): Bbox predictions for each
                batch element (n_q, 10)
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 7) in [cx, cy, cz, l, w, h, theta] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all batch elements of
                    shape (n_q, ).
                - label_weights_list (list[Tensor]): Label weights for all
                    batch elements of shape (n_q, ).
                - bbox_targets_list (list[Tensor]): BBox targets for all
                    batch elements of shape (n_q, 10).
                - bbox_weights_list (list[Tensor]): BBox weights for all
                    batch elements of shape (n_q, 10).
                - num_total_pos (int): Number of positive samples in all \
                    batch elements.
                - num_total_neg (int): Number of negative samples in all \
                    batch elements.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_imgs = len(sem_cls_logits_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            sem_cls_logits_list,
            bbox_pred_list,
            gt_bboxes_list, gt_labels_list,
            gt_bboxes_ignore_list)
        # multi_apply retunrs tuple of lists
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self,
                           sem_cls_logits,
                           bbox_pred,
                           gt_bboxes, gt_labels,
                           gt_bboxes_ignore=None):
        """Compute regression and classification targets for one
            image.
        Args:
            sem_cls_logits (Tensor): Box score logits for each
                batch element (n_q, #cls)
            bbox_pred (Tensor): Bbox predictions for each
                batch element (n_q, 10)
            gt_bboxes (Tensor): Ground truth bboxes for one batch with
                shape (num_gts, 7) in [cx, cy, cz, l, w, h, theta] format.
            gt_labels (Tensor): Ground truth class indices for one batch
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - dir_targets (Tensor): Direction targets for each image.
                - dir_weights (Tensor): Direction weights for each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(bbox_pred,
                                             sem_cls_logits.reshape(-1,
                                                                    self.cls_out_channels),
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds  # (num_pos, ) number of
        # predicted bounding boxes with matched ground truth box, the index
        # of those matched predicted bounding boxes
        neg_inds = sampling_result.neg_inds  # (num_neg, ) indices of
        # negative predicted bounding boxes

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),  # (num_q, ) filled w/ cls
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # here only the matched positive boxes are assigned with labels of
        # matched boxes from ground truth
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # Here, we assign predicted boxes to gt boxes and therefore we want
        # labels and bbox_targets both in predicted box shape but with gt
        # labels and boxes in it!!!
        bbox_targets = torch.zeros_like(bbox_pred)[...,
                       :len(self.code_weights) - 1]
        # because bbox_pred is 10 values but targets is 9 values
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights,
                bbox_targets, bbox_weights,
                pos_inds, neg_inds)
        # labels (num_q, )
        # label_weights (num_q, )
        # bbox_targets (num_q, 7)
        # bbox_weights (num_q, 7)
        # pos_inds (num_q, )
        # neg_inds (num_q, )

    @force_fp32(apply_to='outputs')
    def loss_ota(self, outputs, gt_bboxes_list, gt_labels_list):
        """
        This is the loss for DDet3D

        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes absolute [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            gt_bboxes_list (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            gt_labels_list (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )
        """

        # get gravity center and convert lidar boxes to tensor
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        batch_indices = self.assigner(outputs, gt_bboxes_list,
                                      gt_labels_list, self.num_heads)
        # list[(n_p, )(n_p_gt), ()(), ...bs]
        # fg_mask_inboxes: gives pred indices where it has one matched gt
        # matched_gt_inds: gives the indices of matched gt for each pred box

        # Compute all losses
        loss_cls = self.loss_classification(outputs,
                                            gt_labels_list,
                                            batch_indices)
        loss_bbox = self.loss_boxes(outputs, gt_bboxes_list, batch_indices)

        losses = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox)

        if self.deep_supervision:
            assert 'aux_outputs' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                batch_indices = self.assigner(aux_outputs, gt_bboxes_list,
                                              gt_labels_list, i + 1)
                loss_cls = self.loss_classification(aux_outputs,
                                                    gt_labels_list,
                                                    batch_indices)
                loss_bbox = self.loss_boxes(aux_outputs, gt_bboxes_list,
                                            batch_indices)
                tmp_losses = dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox)
                for name, value in tmp_losses.items():
                    losses[f's.{i}.{name}'] = value
        return losses

    def loss_classification(self, outputs, gt_labels_list, indices):
        """ Classification Loss
        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            gt_labels_list (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )
            indices (list[Tensor]): list[(n_p, )(n_p_gt), ()(), ...bs]
                    fg_mask_inboxes: gives pred indices where it has one matched gt
                    matched_gt_inds: gives the indices of matched gt for each pred box
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (bs, n_p, #cls)
        target_classes_list = [
            gt_label[J] for gt_label, (_, J) in zip(gt_labels_list, indices)
        ]  # [(n_p_gt, ),...]
        target_classes = torch.full(
            src_logits.shape[:2],  # (bs, n_p)
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        for idx in range(len(gt_labels_list)):
            target_classes[idx, indices[idx][0]] = target_classes_list[idx]
            # (bs, n_p)

        src_logits = src_logits.flatten(0, 1)  # (bs*n_p, #cls)
        target_classes = target_classes.flatten(0, 1)  # (bs*n_p, )

        # comp focal loss.
        num_instances = torch.cat(target_classes_list).shape[0]
        if self.sync_cls_avg_factor:
            num_instances = reduce_mean(src_logits.new_tensor([num_instances]))
        num_instances = max(num_instances, 1)
        # bs * n_p_gt
        loss_cls = self.loss_cls(
            src_logits,
            target_classes
        ) / num_instances

        loss_cls = torch.nan_to_num(loss_cls)
        return loss_cls

    def loss_boxes(self, outputs, gt_bboxes_list, indices):
        """ BBox Loss
        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            gt_bboxes_list (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            indices (list[Tensor]): list[(n_p, )(n_p_gt), ()(), ...bs]
                    fg_mask_inboxes: gives pred indices where it has one matched gt
                    matched_gt_inds: gives the indices of matched gt for each pred box
        """
        assert 'pred_boxes' in outputs
        pred_boxes = outputs['pred_boxes']  # (bs, n_p, 10)

        target_bboxes_list = [
            gt_bbox[J] for gt_bbox, (_, J) in zip(gt_bboxes_list, indices)
        ]  # [(n_p_gt, 9),...]

        pred_bboxes_list = []
        for idx in range(len(gt_bboxes_list)):
            pred_bboxes_list.append(pred_boxes[idx, indices[idx][0]])
            # (n_p_gt, 10)

        pred_boxes_cat = torch.cat(pred_bboxes_list)  # (bs*n_p_gt, 10)
        target_bboxes_cat = torch.cat(target_bboxes_list)  # (bs*n_p_gt, 9)

        if len(pred_boxes_cat) > 0:
            num_instances = pred_boxes_cat.shape[0]
            num_instances = pred_boxes_cat.new_tensor([num_instances])
            num_instances = torch.clamp(reduce_mean(num_instances),
                                        min=1).item()

            # for box weights
            bbox_weights = torch.ones_like(pred_boxes_cat)  # (bs*n_p_gt, 10)
            bbox_weights = bbox_weights * self.code_weights  # (bs*n_p_gt, 10)
            # calculate loss between normalized target boxes and pred boxes
            # so pred boes will give normalized result from forward
            normalized_target_bboxes_cat = normalize_bbox(target_bboxes_cat,
                                                          self.pc_range)
            # (bs*n_p_gt, 10)
            isnotnan = torch.isfinite(normalized_target_bboxes_cat).all(dim=-1)
            loss_bbox = self.loss_bbox(
                pred_boxes_cat[isnotnan, :len(self.code_weights)],
                normalized_target_bboxes_cat[
                isnotnan, :len(self.code_weights)], bbox_weights[
                                                    isnotnan,
                                                    :len(
                                                        self.code_weights)]) / num_instances
        else:
            loss_bbox = pred_boxes.sum() * 0

        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_bbox

    def simple_test_bboxes(self, img_feats, point_feats, img_metas):
        """ Test det bboxes without test-time augmentations
        Args:
            img_feats (list[Tensor] | None): Image feats list of stride 4, 8,
                16, 32 of shape (bs, n_cam, C, H, W)
            point_feats (list[Tensor]): Point feat list [(B, 128, H, W)...]
                strides 8, 16, 32, 64 of 1472
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            list[tuple[LiDARBbox, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 9) tensor, where the
                9 columns are bounding box positions
                (cx, cy, cz, l, w, h, theta, vx, vy). The second item is a (n,
                ) tensor where each item is predicted score between 0 and 1.
                The third item is a (n,) tensor where each item is the
                predicted class label of the corresponding box.
        """
        pred_logits, pred_bboxes = self(img_feats, point_feats, img_metas)
        results_list = self.get_bboxes(pred_logits, pred_bboxes, img_metas)
        return results_list

    @force_fp32(apply_to=('pred_logits', 'pred_bboxes',))
    def get_bboxes(self,
                   pred_logits,
                   pred_bboxes,
                   img_metas):
        """ Get the boxes with decoding
        Args:
            pred_logits (Tensor): (#lay, bs, n_p, #cls)
            pred_bboxes (tensor): (#lay, bs, n_p, 10)

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 7) tensor, where the
                7 columns are bounding box positions
                (cx, cy, cz, l, w, h, theta). The second item is a (n,
                ) tensor where each item is predicted score between 0 and 1.
                The third item is a (n,) tensor where each item is the
                predicted class label of the corresponding box.
        """
        pred_logits_lay = pred_logits[-1]
        pred_bboxes_lay = pred_bboxes[-1]

        cfg = self.test_cfg
        cfg = copy.deepcopy(cfg)
        results = []

        if self.use_focal_loss or self.use_fed_loss:
            scores = torch.sigmoid(pred_logits_lay)  # (bs, n_p, #cls)

            # for each sample
            for i, (scores_per_sample,
                    box_pred_per_sample) in enumerate(
                zip(scores, pred_bboxes_lay)):
                # scores_per_sample: (n_p, #cls)
                # box_pred_per_sample: (n_p, 10) ; center:abs and size:log

                # decode the pred boxes: to convert size to exp() and sincos
                # to ry
                box_pred_per_sample = denormalize_bbox(box_pred_per_sample,
                                                       self.pc_range)
                # (n_p, 9) all absolute now
                # convert gravity center to bottom center
                box_pred_per_sample[:, 2] = box_pred_per_sample[:,
                                            2] - box_pred_per_sample[:,
                                                 5] * 0.5
                # (n_p, 9)

                if self.use_nms:
                    # get pred boxes for NMS
                    box_pred_per_sample_for_nms = xywhr2xyxyr(img_metas[i][
                                                                  'box_type_3d'](
                        box_pred_per_sample,
                        box_dim=box_pred_per_sample.shape[
                            -1]).bev)  # (n_p, 5)
                    padding = scores_per_sample.new_zeros(
                        scores_per_sample.shape[0], 1)  # (n_p, 1)
                    # remind that we set FG labels to [0, num_class-1]
                    # BG cat_id: num_class
                    scores_per_sample = torch.cat([scores_per_sample,
                                                   padding], dim=1)
                    # (n_p, #cls + 1)
                    results_nms = box3d_multiclass_nms(box_pred_per_sample,
                                                       box_pred_per_sample_for_nms,
                                                       scores_per_sample,
                                                       cfg.score_thr,
                                                       cfg.max_per_img,
                                                       cfg)

                    box_pred_per_sample, scores_per_sample, \
                        labels_per_sample = results_nms
                    # box_pred (max_num, 9)
                    # scores (max_num, )
                    # labels (max_num, )

                    # check boxes outside of post_center_range
                    post_center_range = torch.tensor(
                        cfg.post_center_range, device=scores_per_sample.device)
                    mask = (box_pred_per_sample[..., :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (box_pred_per_sample[..., :3] <=
                             post_center_range[3:]).all(1)
                    box_pred_per_sample = box_pred_per_sample[mask]
                    scores_per_sample = scores_per_sample[mask]
                    labels_per_sample = labels_per_sample[mask]
                else:
                    scores_per_sample, topk_indices = scores_per_sample.flatten(
                        0, 1).topk(cfg.max_per_img)  # (n_p, )
                    labels_per_sample = topk_indices % self.num_classes
                    # (n_p, )
                    bbox_index = topk_indices // self.num_classes
                    box_pred_per_sample = box_pred_per_sample[bbox_index]
                    # (n_p, 9)

                    # check boxes outside of post_center_range
                    post_center_range = torch.tensor(
                        cfg.post_center_range, device=scores_per_sample.device)
                    mask = (box_pred_per_sample[..., :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (box_pred_per_sample[..., :3] <=
                             post_center_range[3:]).all(1)
                    box_pred_per_sample = box_pred_per_sample[mask]
                    scores_per_sample = scores_per_sample[mask]
                    labels_per_sample = labels_per_sample[mask]

                # convert boxes to LiDARInstanceBbox
                box_pred_per_sample = img_metas[i]['box_type_3d'](
                    box_pred_per_sample, box_pred_per_sample.shape[-1])
                results.append([box_pred_per_sample, scores_per_sample,
                                labels_per_sample])

        else:
            raise NotImplementedError

        return results

    # abstract method loss
    def loss(self, **kwargs):
        pass


@HEADS.register_module()
class SingleSRFDetHeadLiDAR(BaseModule):
    """
    This is the Single SRFDet Head
    """

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 pooler_resolution=7,
                 use_focal_loss=True,
                 use_fed_loss=False,
                 dim_feedforward=2048,
                 num_cls_convs=1,
                 num_reg_convs=3,
                 num_heads=8,
                 dropout=0.0,
                 scale_clamp=_DEFAULT_SCALE_CLAMP,
                 bbox_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2,
                               0.2],
                 act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                 pc_range=None,
                 voxel_size=None,
                 init_cfg=None):
        super(SingleSRFDetHeadLiDAR, self).__init__(init_cfg)

        self.feat_channels_lidar = feat_channels
        self.pc_range_lidar = pc_range
        self.voxel_size_lidar = voxel_size

        # Dynamic
        self.self_attn_lidar = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact_lidar = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1_lidar = nn.Linear(feat_channels, dim_feedforward)
        self.dropout_lidar = nn.Dropout(dropout)
        self.linear2_lidar = nn.Linear(dim_feedforward, feat_channels)

        self.norm1_lidar = nn.LayerNorm(feat_channels)
        self.norm2_lidar = nn.LayerNorm(feat_channels)
        self.norm3_lidar = nn.LayerNorm(feat_channels)
        self.dropout1_lidar = nn.Dropout(dropout)
        self.dropout2_lidar = nn.Dropout(dropout)
        self.dropout3_lidar = nn.Dropout(dropout)

        self.activation_lidar = build_activation_layer(act_cfg)

        # cls.
        cls_module_lidar = list()
        for _ in range(num_cls_convs):
            cls_module_lidar.append(
                nn.Linear(feat_channels, feat_channels, False))
            cls_module_lidar.append(nn.LayerNorm(feat_channels))
            cls_module_lidar.append(nn.ReLU(inplace=True))
        self.cls_module_lidar = ModuleList(cls_module_lidar)

        # reg.
        reg_module_lidar = list()
        for _ in range(num_reg_convs):
            reg_module_lidar.append(
                nn.Linear(feat_channels, feat_channels, False))
            reg_module_lidar.append(nn.LayerNorm(feat_channels))
            reg_module_lidar.append(nn.ReLU(inplace=True))
        self.reg_module_lidar = ModuleList(reg_module_lidar)

        # pred.
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss
        if self.use_focal_loss or self.use_fed_loss:
            self.class_logits_lidar = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits_lidar = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta_lidar = nn.Linear(feat_channels, len(bbox_weights))
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        # for srcn3d
        if init_cfg is None:
            self.init_cfg = [
                dict(
                    type='Normal', std=0.01,
                    override=dict(name='class_logits_lidar')),
                dict(
                    type='Normal', std=0.001,
                    override=dict(name='bboxes_delta_lidar'))
            ]

    def init_weights(self) -> None:
        super(SingleSRFDetHeadLiDAR, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass

        bias_value = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.class_logits_lidar.bias, bias_value)
        nn.init.constant_(self.bboxes_delta_lidar.bias.data[2:], 0.0)

    def forward(self, point_feats, bboxes, prop_feats, pooler, img_metas):
        """
        img_feats (list[Tensor]): shape (bs, n_cam, C, H, W)
        point_feats (list[Tensor]): shape (bs, C, H, W)
        bboxes (Tensor): (bs, n_p, 10)
        prop_feats (Tensor|None): (bs, n_p, dim)
        pooler (nn.Module)
        time_emb (Tensor): (bs, 256*4)
        img_metas (list[dict])
        """

        bs, n_p = bboxes.shape[:2]

        roi_feats_lidar = self.points_feats_sampling_bboxes_roi(
            point_feats,
            bboxes,
            pooler,
            img_metas)
        # (bs*n_p, C, 7, 7)

        if prop_feats is None:
            prop_feats = roi_feats_lidar.view(bs, n_p, self.feat_channels,
                                              -1).mean(-1)
            # (bs, n_p, 256, 7*7) --> (bs, n_p, 256)

        roi_feats = roi_feats_lidar.view(bs * n_p, self.feat_channels_lidar,
                                         -1).permute(2, 0, 1)
        # (bs*n_p, 256, 7*7) --> (7*7, bs*n_p, 256)

        # self_attention
        prop_feats = prop_feats.view(bs, n_p,
                                     self.feat_channels_lidar).permute(
            1, 0, 2)
        # (bs, n_p, 256) --> (n_p, bs, 256)
        prop_feats2 = self.self_attn_lidar(prop_feats, prop_feats,
                                           value=prop_feats)[0]
        prop_feats = prop_feats + self.dropout1_lidar(prop_feats2)
        prop_feats = self.norm1_lidar(prop_feats)  # (n_p, bs, 256)

        # inst_interact.
        prop_feats = prop_feats.view(n_p, bs,
                                     self.feat_channels_lidar).permute(
            1, 0, 2).reshape(1, bs * n_p, self.feat_channels_lidar)
        # (n_p, bs, 256) --> (bs, n_p, 256)  -> (1, bs * n_p, 256)
        # roi feats of shape (7*7, bs*n_p, 256)
        prop_feats2 = self.inst_interact_lidar(prop_feats, roi_feats)
        # (bs*n_p, 256)
        prop_feats = prop_feats + self.dropout2_lidar(prop_feats2)
        obj_feats = self.norm2_lidar(prop_feats)  # (bs*n_p, 256)

        # FFN
        obj_feats2 = self.linear2_lidar(
            self.dropout_lidar(self.activation_lidar(self.linear1_lidar(
                obj_feats))))
        obj_feats = obj_feats + self.dropout3_lidar(obj_feats2)
        obj_feats = self.norm3_lidar(obj_feats)  # (bs*n_p, 256)

        cls_feature = obj_feats.clone()
        reg_feature = obj_feats.clone()
        for cls_layer in self.cls_module_lidar:
            cls_feature = cls_layer(cls_feature)  # (bs*n_p, 256)
        for reg_layer in self.reg_module_lidar:
            reg_feature = reg_layer(reg_feature)  # (bs*n_p, 256)
        class_logits = self.class_logits_lidar(cls_feature)  # (bs*n_p, #cls)
        bboxes_deltas = self.bboxes_delta_lidar(reg_feature)  # (bs*n_p, 10)
        pred_bboxes = self.apply_deltas_lidar(bboxes_deltas, bboxes.view(-1,
                                                                         len(
                                                                             self.bbox_weights)))
        # (bs*n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        return (class_logits.view(bs, n_p, -1), pred_bboxes.view(bs, n_p,
                                                                 -1),
                obj_feats)
        # (bs, n_p, #cls), (bs, n_p, 10), (bs*n_p, 256)
        # pred boxes  center:norm and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    def apply_deltas_lidar(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dz, dw, dl, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*10),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 10)
                [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        """
        boxes = boxes.to(deltas.dtype)
        deltas_split = torch.split(deltas, 1, dim=-1)
        # ((bs, n_p, 1), ... 10)
        boxes_split = torch.split(boxes, 1, dim=-1)
        # ((bs, n_p, 1), ... 10)
        if len(self.bbox_weights) == 10:
            wx, wy, wz, ww, wl, wh, _, _, _, _ = self.bbox_weights
        else:
            wx, wy, wz, ww, wl, wh, _, _ = self.bbox_weights

        dx = deltas_split[0] / wx
        dy = deltas_split[1] / wy
        dz = deltas_split[2] / wz
        dw = deltas_split[3] / ww
        dl = deltas_split[4] / wl
        dh = deltas_split[5] / wh

        ctr_x = boxes_split[0]
        ctr_y = boxes_split[1]
        ctr_z = boxes_split[2]
        # ctr_x = boxes_split[0] * (self.pc_range[3] - self.pc_range[0]) + \
        #     self.pc_range[0]
        # ctr_y = boxes_split[1] * (self.pc_range[4] - self.pc_range[1]) + \
        #         self.pc_range[1]
        # ctr_z = boxes_split[2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        widths = torch.exp(boxes_split[3])
        lengths = torch.exp(boxes_split[4])
        heights = torch.exp(boxes_split[5])
        # because log is applied at end so to reverse it
        # widths = boxes_split[3]
        # lengths = boxes_split[4]
        # heights = boxes_split[5]

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dl = torch.clamp(dl, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * lengths + ctr_y
        pred_ctr_z = dz * heights + ctr_z

        pred_w = torch.exp(dw) * widths
        pred_l = torch.exp(dl) * lengths
        pred_h = torch.exp(dh) * heights  # (bs*n_p, 1)

        pred_ctr_x = (pred_ctr_x - self.pc_range_lidar[0]) / (
                self.pc_range_lidar[3] -
                self.pc_range_lidar[0])
        pred_ctr_y = (pred_ctr_y - self.pc_range_lidar[1]) / (
                self.pc_range_lidar[4] -
                self.pc_range_lidar[1])
        pred_ctr_z = (pred_ctr_z - self.pc_range_lidar[2]) / (
                self.pc_range_lidar[5] -
                self.pc_range_lidar[2])
        pred_ctr_x = torch.clamp(pred_ctr_x, max=1.0, min=0.0)
        pred_ctr_y = torch.clamp(pred_ctr_y, max=1.0, min=0.0)
        pred_ctr_z = torch.clamp(pred_ctr_z, max=1.0, min=0.0)

        if len(self.bbox_weights) == 10:
            pred_boxes = torch.cat(
                [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w.log(),
                 pred_l.log(),
                 pred_h.log(), deltas_split[6], deltas_split[7],
                 deltas_split[8],
                 deltas_split[9]], dim=-1)  # (bs*n_p, 10)
        else:
            pred_boxes = torch.cat(
                [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w.log(),
                 pred_l.log(),
                 pred_h.log(), deltas_split[6], deltas_split[7]],
                dim=-1)  # (bs*n_p, 10)
        # pred_boxes = torch.cat(
        #     [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w, pred_l,
        #      pred_h, deltas_split[6], deltas_split[7], deltas_split[8],
        #      deltas_split[9]], dim=-1)  # (bs*n_p, 10)

        return pred_boxes  # (bs*n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    def points_feats_sampling_bboxes_roi(self, points_feats, bboxes, pooler,
                                         img_metas):
        """
        This function samples the LiDAR features for the bboxes using pooler
        points_feats (list[Tensor]): shape (bs, 256, H, W) stride 8, 16, 32, 64
        bboxes (Tensor): (bs, n_p, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        pooler (nn.Module): ROI Extractor
        img_metas (list[dict])
        pc_range (list): [x_min, y_,in, z_min, x_max, y_max, z_max]
        """

        # convert box center to world coordinates
        pc_range_ = bboxes.new_tensor(
            [[self.pc_range_lidar[3] - self.pc_range_lidar[0],
              self.pc_range_lidar[4] - self.pc_range_lidar[1],
              self.pc_range_lidar[5] - self.pc_range_lidar[2]]])  # (1, 3)
        pc_start_ = bboxes.new_tensor(
            [[self.pc_range_lidar[0], self.pc_range_lidar[1],
              self.pc_range_lidar[2]]])  # (1, 3)
        bboxes[..., :3] = (bboxes[..., :3] * pc_range_) + pc_start_  # (n_p, 3)

        # get the corners of the bboxes
        bbox_corners = boxes3d_to_corners3d(bboxes[..., :8],
                                            bottom_center=False, ry=False)
        # (bs, n_p, 8, 3) in world coord

        # convert corners to range [0, 110.4]
        pc_start_ = bboxes.new_tensor(
            [[self.pc_range_lidar[0], self.pc_range_lidar[1],
              self.pc_range_lidar[2]]])  # (1, 3)
        bbox_corners = bbox_corners - pc_start_  # (bs, n_p, 8, 3)

        # divide by voxel size to get index on BEV
        bbox_corners[..., 0:1] = bbox_corners[..., 0:1] / \
                                 self.voxel_size_lidar[0]
        bbox_corners[..., 1:2] = bbox_corners[..., 1:2] / \
                                 self.voxel_size_lidar[1]
        # range [0, 1472]

        bbox_corners_bev = bbox_corners[..., :2]  # (bs, n_p, 8, 2)

        # expect box_corners_in_bev: [B,N, 8, 2] -- [B,num_cam,N,8,2]
        minxy = torch.min(bbox_corners_bev, dim=2).values
        # (bs, n_p, 2)
        maxxy = torch.max(bbox_corners_bev, dim=2).values
        # (bs, n_p, 2)

        bbox2d = torch.cat([minxy, maxxy], dim=2)
        # (bs, n_p, 4)

        # convert bbox2d to ROI
        bbox2d_list = torch.split(bbox2d, 1)
        # ((1, n_p, 4), (1, n_p, 4),...bs)
        bbox2d_list = [lvl[0, :, :] for lvl in bbox2d_list]
        # [(n_p, 4), (n_p, 4), ... bs]

        rois = bbox2roi(bbox2d_list)  # (bs*n_p, 5) batch_id

        sampled_feats = pooler(points_feats[:pooler.num_inputs], rois)
        # (bs*n_p, C, 7, 7)

        return sampled_feats
        # (bs*n_p, C, 7, 7)


@HEADS.register_module()
class SingleSRFDetHeadImg(BaseModule):
    """
    This is the Single SRFDet Head
    """

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 pooler_resolution=7,
                 use_focal_loss=True,
                 use_fed_loss=False,
                 dim_feedforward=2048,
                 num_cls_convs=1,
                 num_reg_convs=3,
                 num_heads=8,
                 dropout=0.0,
                 scale_clamp=_DEFAULT_SCALE_CLAMP,
                 bbox_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2,
                               0.2],
                 act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                 pc_range=None,
                 init_cfg=None):
        super(SingleSRFDetHeadImg, self).__init__(init_cfg)

        self.feat_channels_img = feat_channels
        self.pc_range_img = pc_range

        # Dynamic
        self.self_attn_img = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact_img = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1_img = nn.Linear(feat_channels, dim_feedforward)
        self.dropout_img = nn.Dropout(dropout)
        self.linear2_img = nn.Linear(dim_feedforward, feat_channels)

        self.norm1_img = nn.LayerNorm(feat_channels)
        self.norm2_img = nn.LayerNorm(feat_channels)
        self.norm3_img = nn.LayerNorm(feat_channels)
        self.dropout1_img = nn.Dropout(dropout)
        self.dropout2_img = nn.Dropout(dropout)
        self.dropout3_img = nn.Dropout(dropout)

        self.activation_img = build_activation_layer(act_cfg)

        # cls.
        cls_module_img = list()
        for _ in range(num_cls_convs):
            cls_module_img.append(
                nn.Linear(feat_channels, feat_channels, False))
            cls_module_img.append(nn.LayerNorm(feat_channels))
            cls_module_img.append(nn.ReLU(inplace=True))
        self.cls_module_img = ModuleList(cls_module_img)

        # reg.
        reg_module_img = list()
        for _ in range(num_reg_convs):
            reg_module_img.append(
                nn.Linear(feat_channels, feat_channels, False))
            reg_module_img.append(nn.LayerNorm(feat_channels))
            reg_module_img.append(nn.ReLU(inplace=True))
        self.reg_module_img = ModuleList(reg_module_img)

        # pred.
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss
        if self.use_focal_loss or self.use_fed_loss:
            self.class_logits_img = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits_img = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta_img = nn.Linear(feat_channels, len(bbox_weights))
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        if init_cfg is None:
            self.init_cfg = [
                dict(
                    type='Normal', std=0.01,
                    override=dict(name='class_logits_img')),
                dict(
                    type='Normal', std=0.001,
                    override=dict(name='bboxes_delta_img'))
            ]

    def init_weights(self) -> None:
        super(SingleSRFDetHeadImg, self).init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass

        bias_value = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.class_logits_img.bias, bias_value)
        nn.init.constant_(self.bboxes_delta_img.bias.data[2:], 0.0)

    def forward(self, img_feats, bboxes, prop_feats, pooler, img_metas):
        """
        img_feats (list[Tensor]): shape (bs, n_cam, C, H, W)
        bboxes (Tensor): (bs, n_p, 10)
        prop_feats (Tensor|None): (bs, n_p, dim)
        pooler (nn.Module)
        time_emb (Tensor): (bs, 256*4)
        img_metas (list[dict])
        """

        bs, n_p = bboxes.shape[:2]

        roi_feats_img = self.img_feats_sampling_bboxes_roi(img_feats,
                                                           bboxes,
                                                           pooler,
                                                           img_metas)
        # (bs*n_p, C, 7, 7)

        if prop_feats is None:
            prop_feats = roi_feats_img.view(bs, n_p, self.feat_channels,
                                            -1).mean(-1)
            # (bs, n_p, 256, 7*7) --> (bs, n_p, 256)

        roi_feats = roi_feats_img.view(bs * n_p, self.feat_channels_img,
                                       -1).permute(2, 0, 1)
        # (bs*n_p, 256, 7*7) --> (7*7, bs*n_p, 256)

        # self_attention
        prop_feats = prop_feats.view(bs, n_p, self.feat_channels_img).permute(
            1, 0, 2)
        # (bs, n_p, 256) --> (n_p, bs, 256)
        prop_feats2 = self.self_attn_img(prop_feats, prop_feats,
                                         value=prop_feats)[0]
        prop_feats = prop_feats + self.dropout1_img(prop_feats2)
        prop_feats = self.norm1_img(prop_feats)  # (n_p, bs, 256)

        # inst_interact.
        prop_feats = prop_feats.view(n_p, bs, self.feat_channels_img).permute(
            1, 0, 2).reshape(1, bs * n_p, self.feat_channels_img)
        # (n_p, bs, 256) --> (bs, n_p, 256)  -> (1, bs * n_p, 256)
        # roi feats of shape (7*7, bs*n_p, 256)
        prop_feats2 = self.inst_interact_img(prop_feats, roi_feats)
        # (bs*n_p, 256)
        prop_feats = prop_feats + self.dropout2_img(prop_feats2)
        obj_feats = self.norm2_img(prop_feats)  # (bs*n_p, 256)

        # FFN
        obj_feats2 = self.linear2_img(
            self.dropout_img(self.activation_img(self.linear1_img(obj_feats))))
        obj_feats = obj_feats + self.dropout3_img(obj_feats2)
        obj_feats = self.norm3_img(obj_feats)  # (bs*n_p, 256)

        cls_feature = obj_feats.clone()
        reg_feature = obj_feats.clone()
        for cls_layer in self.cls_module_img:
            cls_feature = cls_layer(cls_feature)  # (bs*n_p, 256)
        for reg_layer in self.reg_module_img:
            reg_feature = reg_layer(reg_feature)  # (bs*n_p, 256)
        class_logits = self.class_logits_img(cls_feature)  # (bs*n_p, #cls)
        bboxes_deltas = self.bboxes_delta_img(reg_feature)  # (bs*n_p, 10)
        pred_bboxes = self.apply_deltas_img(bboxes_deltas, bboxes.view(-1,
                                                                       len(
                                                                           self.bbox_weights)))
        # (bs*n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        return (class_logits.view(bs, n_p, -1), pred_bboxes.view(bs, n_p,
                                                                 -1),
                obj_feats)
        # (bs, n_p, #cls), (bs, n_p, 10), (bs*n_p, 256)
        # pred boxes  center:norm and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    def apply_deltas_img(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dz, dw, dl, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*10),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 10)
                [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        """
        boxes = boxes.to(deltas.dtype)
        deltas_split = torch.split(deltas, 1, dim=-1)
        # ((bs, n_p, 1), ... 10)
        boxes_split = torch.split(boxes, 1, dim=-1)
        # ((bs, n_p, 1), ... 10)
        if len(self.bbox_weights) == 10:
            wx, wy, wz, ww, wl, wh, _, _, _, _ = self.bbox_weights
        else:
            wx, wy, wz, ww, wl, wh, _, _ = self.bbox_weights

        dx = deltas_split[0] / wx
        dy = deltas_split[1] / wy
        dz = deltas_split[2] / wz
        dw = deltas_split[3] / ww
        dl = deltas_split[4] / wl
        dh = deltas_split[5] / wh

        ctr_x = boxes_split[0]
        ctr_y = boxes_split[1]
        ctr_z = boxes_split[2]
        # ctr_x = boxes_split[0] * (self.pc_range[3] - self.pc_range[0]) + \
        #     self.pc_range[0]
        # ctr_y = boxes_split[1] * (self.pc_range[4] - self.pc_range[1]) + \
        #         self.pc_range[1]
        # ctr_z = boxes_split[2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        widths = torch.exp(boxes_split[3])
        lengths = torch.exp(boxes_split[4])
        heights = torch.exp(boxes_split[5])
        # because log is applied at end so to reverse it
        # widths = boxes_split[3]
        # lengths = boxes_split[4]
        # heights = boxes_split[5]

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dl = torch.clamp(dl, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * lengths + ctr_y
        pred_ctr_z = dz * heights + ctr_z

        pred_w = torch.exp(dw) * widths
        pred_l = torch.exp(dl) * lengths
        pred_h = torch.exp(dh) * heights  # (bs*n_p, 1)

        pred_ctr_x = (pred_ctr_x - self.pc_range_img[0]) / (
                self.pc_range_img[3] -
                self.pc_range_img[0])
        pred_ctr_y = (pred_ctr_y - self.pc_range_img[1]) / (
                self.pc_range_img[4] -
                self.pc_range_img[1])
        pred_ctr_z = (pred_ctr_z - self.pc_range_img[2]) / (
                self.pc_range_img[5] -
                self.pc_range_img[2])
        pred_ctr_x = torch.clamp(pred_ctr_x, max=1.0, min=0.0)
        pred_ctr_y = torch.clamp(pred_ctr_y, max=1.0, min=0.0)
        pred_ctr_z = torch.clamp(pred_ctr_z, max=1.0, min=0.0)

        if len(self.bbox_weights) == 10:
            pred_boxes = torch.cat(
                [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w.log(),
                 pred_l.log(),
                 pred_h.log(), deltas_split[6], deltas_split[7],
                 deltas_split[8],
                 deltas_split[9]], dim=-1)  # (bs*n_p, 10)
        else:
            pred_boxes = torch.cat(
                [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w.log(),
                 pred_l.log(),
                 pred_h.log(), deltas_split[6], deltas_split[7]],
                dim=-1)  # (bs*n_p, 10)
        # pred_boxes = torch.cat(
        #     [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w, pred_l,
        #      pred_h, deltas_split[6], deltas_split[7], deltas_split[8],
        #      deltas_split[9]], dim=-1)  # (bs*n_p, 10)

        return pred_boxes  # (bs*n_p, 10)
        # pred boxes  center:norm and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    def img_feats_sampling_bboxes_roi(self, img_feats, bboxes, pooler,
                                      img_metas):
        """
        This function samples the image features for the bboxes using pooler
        img_feats (list[Tensor]): shape (bs, n_cam, C, H, W)
        bboxes (Tensor): (bs, n_p, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        pooler (nn.Module): ROI Extractor
        img_metas (list[dict])
        pc_range (list): [x_min, y_,in, z_min, x_max, y_max, z_max]
        """

        # convert box center to world coordinates
        pc_range_ = bboxes.new_tensor(
            [[self.pc_range_img[3] - self.pc_range_img[0],
              self.pc_range_img[4] - self.pc_range_img[1],
              self.pc_range_img[5] - self.pc_range_img[2]]])  # (1, 3)
        pc_start_ = bboxes.new_tensor(
            [[self.pc_range_img[0], self.pc_range_img[1],
              self.pc_range_img[2]]])  # (1, 3)
        bboxes[..., :3] = (bboxes[..., :3] * pc_range_) + pc_start_  # (n_p, 3)

        # get the corners of the bboxes
        bbox_corners = boxes3d_to_corners3d(bboxes[..., :8],
                                            bottom_center=False, ry=False)
        # (bs, n_p, 8, 3) in world coord

        # project the corners in LiDAR coord system to six cameras
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = bboxes.new_tensor(lidar2img)  # (bs, num_cam, 4, 4)
        bbox_corners = torch.cat((bbox_corners,
                                  torch.ones_like(
                                      bbox_corners[..., :1])), -1)
        # (bs, n_p, 8, 4)
        bs, num_prop = bbox_corners.size()[:2]
        num_cam = lidar2img.size(1)
        bbox_corners = bbox_corners.view(bs, 1, num_prop, 8, 4).repeat(1,
                                                                       num_cam,
                                                                       1,
                                                                       1,
                                                                       1).unsqueeze(
            -1)
        # (bs, n_p, 8, 4) --> (bs, 1, n_p, 8, 4) --> (bs, n_cam, n_p, 8, 4) -->
        # (bs, n_cam, n_p, 8, 4, 1)
        lidar2img = lidar2img.view(bs, num_cam, 1, 1, 4, 4).repeat(1, 1,
                                                                   num_prop,
                                                                   8, 1, 1)
        # (bs, num_cam, 4, 4) --> (bs, n_cam, 1, 1, 4, 4) --> (bs, n_cam, n_p,
        # 8, 4, 4)

        bbox_cam = torch.matmul(lidar2img, bbox_corners).squeeze(-1)
        # (bs, num_cam, num_proposals, 8, 4)
        # (bs, n_cam, n_p, 8, 4, 4) * (bs, n_cam, n_p, 8, 4, 1) =
        # (bs, n_cam, n_p, 8, 4, 1) --> (bs, n_cam, n_p, 8, 4)

        # normalize real-world points back to normalized [-1,-1,1,1]
        # image coordinate
        eps = 1e-5
        bbox_cam = bbox_cam[..., 0:2] / torch.maximum(bbox_cam[..., 2:3],
                                                      torch.ones_like(
                                                          bbox_cam[...,
                                                          2:3]) * eps)  # ?
        # (bs, n_cam, n_p, 8, 2)

        box_corners_in_image = bbox_cam

        # expect box_corners_in_image: [B,N, 8, 2] -- [B,num_cam,N,8,2]
        minxy = torch.min(box_corners_in_image, dim=3).values
        # (bs, n_cam, n_p, 2)
        maxxy = torch.max(box_corners_in_image, dim=3).values
        # (bs, n_cam, n_p, 2)

        bbox2d = torch.cat([minxy, maxxy], dim=3).permute(0, 2, 1, 3)
        # (bs, n_cam, n_p, 4) --> (bs, n_p, n_cam, 4)

        # convert bbox2d to ROI for all cameras
        sampled_rois = None
        for cam_id in range(num_cam):
            bs = img_feats[0].shape[0]
            C = img_feats[0].shape[2]

            bbox2d_percam = bbox2d[:, :, cam_id, :].reshape(bs, num_prop, 4)
            # (bs, n_p, 4)

            bbox2d_percam_list = torch.split(bbox2d_percam, 1)
            # ((1, n_p, 4), (1, n_p, 4),...bs)
            bbox2d_percam_list = [lvl[0, :, :] for lvl in bbox2d_percam_list]
            # [(n_p, 4), (n_p, 4), ... bs]

            if sampled_rois is None:
                temp_roi = bbox2roi(bbox2d_percam_list)  # (bs*n_p, 5) batch_id
                temp_roi[:, 0] = temp_roi[:, 0] + cam_id * bs
                # batch and cam ids
                sampled_rois = temp_roi
            else:
                temp_roi = bbox2roi(bbox2d_percam_list)
                temp_roi[:, 0] = temp_roi[:, 0] + cam_id * bs
                sampled_rois = torch.cat([sampled_rois, temp_roi], dim=0)
                # (bs*n_p*n_cam, 5)

                # here for bs=3, n_p=4 and n_cam=6, the temp_roi[:, 0] is...
                # cam1: 0000,1111,2222
                # cam2: 3333,4444,5555
                # cam3: 6666,7777,8888
                # cam4: 9999, 10101010, 11111111 and so on

                # img_feat_lvl is (bs, n_cam, C, H, W)
                # but extarctor expects it to be (bs, C, H, W)
                # so lets permute img_feat_lvl to (bs*n_cam, C, H, W)
                # here the 1st dimenion values would be
                # 0, 1, 2, 3, 4, 5, ... 18

        # mlvl_feats_cam = [feat[0, :, :, :, :] for feat in img_feats]
        mlvl_feats_cam = []
        for feat in img_feats:
            bs, n_cam, C, H, W = feat.shape
            mlvl_feats_cam.append(feat.reshape(bs*n_cam, C, H, W))
        sampled_feats = pooler(mlvl_feats_cam[:pooler.num_inputs],
                               sampled_rois)
        # (num_cam * num_prop, C, 7, 7)
        sampled_feats = sampled_feats.view(num_cam, bs, num_prop, C, 7, 7)
        # (n_cam, bs, n_p, C, 7, 7)
        sampled_feats = sampled_feats.permute(1, 0, 2, 3, 4, 5)
        # (bs, n_cam, n_p, C, 7, 7)
        sampled_feats = sampled_feats.permute(0, 2, 3, 1, 4, 5)
        # (bs, n_cam, n_p, C, 7, 7) --> (bs, n_p, C, n_cam, 7, 7)
        sampled_feats = sampled_feats.reshape(bs, num_prop, C, num_cam, 7, 7)
        sampled_feats = sampled_feats.permute(0, 1, 2, 4, 5, 3)
        # (bs, n_p, C, n_cam, 7, 7) --> (bs, n_p, C, 7, 7, n_cam)

        sampled_feats = sampled_feats.sum(-1)  # (bs, n_p, C, 7, 7)
        sampled_feats = sampled_feats.view(bs * num_prop, C, 7, 7)
        # (bs*n_p, C, 7, 7)

        return sampled_feats
        # (bs*n_p, C, 7, 7)


class DynamicConv(nn.Module):

    def __init__(self,
                 feat_channels: int,
                 dynamic_dim: int = 64,
                 dynamic_num: int = 2,
                 pooler_resolution: int = 7) -> None:
        super().__init__()

        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim
        self.dynamic_layer = nn.Linear(self.feat_channels,
                                       self.dynamic_num * self.num_params)

        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.feat_channels * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, prop_feats, roi_feats):
        """Forward function.

        Args:
            prop_feats: (1,  bs * n_p, C)
            roi_feats: (7*7, bs * n_p, C)

        Returns:
        """
        features = roi_feats.permute(1, 0, 2)  # (bs*n_p, 7*7, C)
        parameters = self.dynamic_layer(prop_feats).permute(1, 0, 2)
        # (1, bs * n_p, 2 * C * C/4) --> (bs * n_p, 1, 2 * C * C/4)

        param1 = parameters[:, :, :self.num_params].view(
            -1, self.feat_channels, self.dynamic_dim)
        # (bs*n_p, C, C/4)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dynamic_dim,
                                                         self.feat_channels)
        # (bs*n_p, C/4, C)

        # (bs*n_p, 7*7, C) * (bs*n_p, C, C/4) = (bs*n_p, 7*7, C/4)
        features = torch.bmm(features, param1)  # (bs*n_p, 7*7, C/4)
        features = self.norm1(features)
        features = self.activation(features)

        # (bs*n_p, 7*7, C/4) * (bs*n_p, C/4, C) = (bs*n_p, 7*7, C)
        features = torch.bmm(features, param2)  # (bs*n_p, 7*7, C)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)  # (bs*n_p, 7*7*C)
        features = self.out_layer(features)  # (bs*n_p, C)
        features = self.norm3(features)
        features = self.activation(features)

        return features  # (bs*n_p, C)
