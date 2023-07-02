import numpy as np
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.datasets import KittiDataset
from mmdet3d.core.bbox import Box3DMode, Coord3DMode

from ..core.visualizer import (show_result, show_multi_modality_result,
                               show_bev_result_kitti)


@DATASETS.register_module()
class CustomKittiDataset(KittiDataset):

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
            mask = gt_bboxes.in_range_bev([0, -40, 70.4, 40])
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
            show_gt_bboxes = Box3DMode.convert(gt_bboxes_numpy, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)

            # Get Prediction Boxes
            inds = result['scores_3d'] > show_threshold
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_numpy = pred_bboxes.tensor.numpy()
            pred_labels = result['labels_3d'][inds]
            show_pred_bboxes = Box3DMode.convert(pred_bboxes_numpy, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            # show_result(points_depth, show_gt_bboxes, show_pred_bboxes,
            #             out_dir,
            #             file_name, show, pred_labels=pred_labels,
            #             gt_labels=gt_labels)

            # BEV Show and Save
            show_bev_result_kitti(points_org, coord_type='LIDAR',
                            gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes,
                            out_dir=out_dir, filename=str(i), show=show,
                            pred_labels=pred_labels, gt_labels=gt_labels,
                            save=save_imgs, voxel_size=0.1, bev_img_size=1024)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_multi_modality_result(
                    img,
                    gt_bboxes,
                    pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    str(i),
                    box_mode='lidar',
                    show=False,
                    pred_labels=pred_labels,
                    gt_labels=gt_labels,
                    view=str(i),
                    save=save_imgs
                )
