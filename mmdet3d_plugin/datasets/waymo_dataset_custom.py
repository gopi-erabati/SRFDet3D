import numpy as np
from os import path as osp
import os

from mmdet.datasets import DATASETS
from mmdet3d.datasets import WaymoDataset
from mmdet3d.core.bbox import Box3DMode, Coord3DMode

from ..core.visualizer import (show_result, show_multi_modality_result,
                               show_bev_result_waymo)


@DATASETS.register_module()
class CustomWaymoDataset(WaymoDataset):
    """ This dataset customizes for multi-view images in Waymo
    """

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 num_views=5,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 **kwargs
                 ):
        super().__init__(data_root=data_root,
                         ann_file=ann_file,
                         split=split,
                         pts_prefix=pts_prefix,
                         pipeline=pipeline,
                         classes=classes,
                         modality=modality,
                         box_type_3d=box_type_3d,
                         filter_empty_gt=filter_empty_gt,
                         test_mode=test_mode,
                         load_interval=load_interval,
                         pcd_limit_range=pcd_limit_range,
                         **kwargs)
        self.num_views = num_views

    # def get_data_info(self, index):
    #     """Get data info according to the given index.
    #
    #             Args:
    #                 index (int): Index of the sample data to get.
    #
    #             Returns:
    #                 dict: Standard input_dict consists of the
    #                     data information.
    #
    #                     - sample_idx (str): sample index
    #                     - pts_filename (str): filename of point clouds
    #                     - img_prefix (str): prefix of image files
    #                     - img_info (dict): image info
    #                     - lidar2img (list[np.ndarray], optional): transformations from
    #                         lidar to different cameras
    #                     - ann_info (dict): annotation info
    #             """
    #     info = self.data_infos[index]
    #     sample_idx = info['image']['image_idx']
    #     img_filename = os.path.join(self.data_root,
    #                                 info['image']['image_path'])
    #
    #     # TODO: consider use torch.Tensor only
    #     rect = info['calib']['R0_rect'].astype(np.float32)
    #     Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
    #     P0 = info['calib']['P0'].astype(np.float32)
    #     lidar2img = P0 @ rect @ Trv2c
    #
    #     # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
    #     # the size of img0-2: 1280x1920; img3-4: 886x1920
    #     if self.modality['use_camera']:
    #         image_paths = []
    #         lidar2img_rts = []
    #
    #         # load calibration for all 5 images.
    #         calib_path = img_filename.replace('image_0', 'calib').replace(
    #             '.png', '.txt').replace('.jpg', '.txt')
    #         Tr_velo_to_cam_list = []
    #         with open(calib_path, 'r') as f:
    #             lines = f.readlines()
    #         for line_num in range(6, 6 + self.num_views):
    #             trans = np.array([float(info) for info in
    #                               lines[line_num].split(' ')[1:13]]).reshape(3,
    #                                                                          4)
    #             trans = np.concatenate([trans, np.array([[0., 0., 0., 1.]])],
    #                                    axis=0).astype(np.float32)
    #             Tr_velo_to_cam_list.append(trans)
    #         assert np.allclose(Tr_velo_to_cam_list[0],
    #                            info['calib']['Tr_velo_to_cam'].astype(
    #                                np.float32))
    #
    #         for idx_img in range(self.num_views):
    #             rect = info['calib']['R0_rect'].astype(np.float32)
    #             # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
    #             Trv2c = Tr_velo_to_cam_list[idx_img]
    #             P0 = info['calib'][f'P{idx_img}'].astype(np.float32)
    #             lidar2img = P0 @ rect @ Trv2c
    #
    #             image_paths.append(
    #                 img_filename.replace('image_0', f'image_{idx_img}'))
    #             lidar2img_rts.append(lidar2img)
    #
    #     pts_filename = self._get_pts_filename(sample_idx)
    #     input_dict = dict(
    #         sample_idx=sample_idx,
    #         pts_filename=pts_filename,
    #         img_prefix=None, )
    #     if self.modality['use_camera']:
    #         input_dict['img_filename'] = image_paths
    #         input_dict['lidar2img'] = lidar2img_rts
    #
    #     if not self.test_mode:
    #         annos = self.get_ann_info(index)
    #         input_dict['ann_info'] = annos
    #
    #     return input_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        show_threshold = 0.2
        save_imgs = True
        from tqdm import tqdm
        for i, result in tqdm(enumerate(results)):
            if i not in [423, 448, 525, 685, 1000, 1100,
                         2600, 2726, 2981]:
                continue
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points_org = points.numpy()

            # Show boxes on point cloud
            # for now we convert points into depth mode
            points_depth = Coord3DMode.convert_point(points_org,
                                                     Coord3DMode.LIDAR,
                                                     Coord3DMode.DEPTH)

            # Get GT Boxes and Filter by range and name
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            mask = gt_bboxes.in_range_bev([-76.8, -76.8, 76.8, 76.8])
            gt_bboxes = gt_bboxes[mask]
            gt_bboxes.limit_yaw(offset=0.5, period=2 * np.pi)
            gt_labels = gt_labels[mask.numpy().astype(np.bool)]
            # name filtering
            labels = list(range(3))
            gt_bboxes_mask = np.array([n in labels for n in gt_labels],
                                      dtype=np.bool_)
            gt_bboxes = gt_bboxes[gt_bboxes_mask]
            gt_labels = gt_labels[gt_bboxes_mask]

            # Convert GT boxes to Depth Mode to show with Visualizer
            gt_bboxes_numpy = gt_bboxes.tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes_numpy,
                                               Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)

            # Get Prediction Boxes
            inds = result['scores_3d'] > show_threshold
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_numpy = pred_bboxes.tensor.numpy()
            pred_labels = result['labels_3d'][inds]
            show_pred_bboxes = Box3DMode.convert(pred_bboxes_numpy,
                                                 Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points_depth, show_gt_bboxes, show_pred_bboxes,
                        out_dir,
                        file_name, show, pred_labels=pred_labels,
                        gt_labels=gt_labels)

            # BEV Show and Save
            # show_bev_result_waymo(points_org, coord_type='LIDAR',
            #                       gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes,
            #                       out_dir=out_dir, filename=str(i), show=show,
            #                       pred_labels=pred_labels, gt_labels=gt_labels,
            #                       save=save_imgs, voxel_size=0.2,
            #                       bev_img_size=1024)
