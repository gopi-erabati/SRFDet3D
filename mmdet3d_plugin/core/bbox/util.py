import torch


def normalize_bbox(bboxes, pc_range):
    """ Normalize the bbox
    Normalize the center, log the size, sincos the rot

    Args:
        bboxes (Tensor): (n_p, 9) with vx and vy (or) (n_p, 7)

    Returns:
         Tensor of shape (n_p, 10) or (n_p, 8)
    """
    center = bboxes[..., 0:3]  # (n_p, 3)

    # # Normalize center to [0, 1]
    # # center normalize
    # pc_range_ = bboxes.new_tensor([[pc_range[3] - pc_range[0],
    #                                 pc_range[4] - pc_range[1],
    #                                 pc_range[5] - pc_range[2]]])  # (1, 3)
    # pc_start_ = bboxes.new_tensor(
    #     [[pc_range[0], pc_range[1], pc_range[2]]])  # (1, 3)
    # bbox_center = (center - pc_start_) / pc_range_  # (n_p, 3)

    size = bboxes[..., 3:6].log()  # (n_p, 3)

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (center, size, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (center, size, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes  # (n_p, 10) or (n_p, 8)


def denormalize_bbox(normalized_bboxes, pc_range):
    """ Denormlaize the bbox
    Denormalize the center, exp the size, convert sincos to rot

    Args:
        normalized_bboxes (Tensor): (n_p, 10) with vx and vy (or) (n_p, 8)
        pc_range (list): [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
         Tensor of shape (n_p, 9) or (n_p, 7)
    """

    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center
    center = normalized_bboxes[..., 0:3]  # (n_p, 3)
    # # center denormalize
    # pc_range_ = normalized_bboxes.new_tensor([[pc_range[3] - pc_range[0],
    #                                            pc_range[4] - pc_range[1],
    #                                            pc_range[5] - pc_range[
    #                                                2]]])  # (1, 3)
    # pc_start_ = normalized_bboxes.new_tensor(
    #     [[pc_range[0], pc_range[1], pc_range[2]]])  # (1, 3)
    # bbox_center = (center * pc_range_) + pc_start_  # (n_p, 3)

    # size
    size = normalized_bboxes[..., 3:6].exp()  # (n_p, 3)

    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat([center, size, rot, vx, vy],
                                        dim=-1)
    else:
        denormalized_bboxes = torch.cat([center, size, rot], dim=-1)
    return denormalized_bboxes  # (n_p, 9) or (n_p, 7)


def boxes3d_to_corners3d(boxes3d, bottom_center=True, ry=False):
    """Convert kitti center boxes to corners.

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        boxes3d (torch.tensor): Boxes with shape of (bs, N, 8) if ry=True
            or (bs, N, 7)
            cx, cy, cz, w, l,  h, sin(rot), cos(rot) in TOP LiDAR coords,
            see the definition of ry in nuScenes dataset.
        bottom_center (bool, optional): Whether z is on the bottom center
            of object. Defaults to True.
        ry (bool, optional): whether angle in ry or sincos format

    Returns:
        torch.tensor: Box corners with the shape of [bs,N, 8, 3].
    """

    bs = boxes3d.shape[0]
    boxes_num = boxes3d.shape[1]

    if ry:
        cx, cy, cz, w, l, h, ry = tuple(
            [boxes3d[:, :, i] for i
             in range(boxes3d.shape[2])])
    else:
        cx, cy, cz, w, l, h, sin_rot, cos_rot = tuple(
            [boxes3d[:, :, i] for i
             in range(boxes3d.shape[2])])
        # (bs, n_p)
        ry = torch.atan2(sin_rot.clone(), cos_rot.clone())  # (bs, n_p)

    w = w.exp()
    l = l.exp()
    h = h.exp()

    # w, l, h: (B,N)
    x_corners = torch.stack(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dim=2)  # (B,N,8)
    y_corners = torch.stack(
        [-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.],
        dim=2)  # .T
    if bottom_center:
        z_corners = torch.zeros((bs, boxes_num, 8), dtype=torch.float32).cuda()
        z_corners[:, :, 4:8] = torch.unsqueeze(h, 2).expand(bs, boxes_num,
                                                            4)  # (bs, N, 8)
    else:
        z_corners = torch.stack([
            -h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.
        ], dim=2)  # .T

    # ry = rot # (bs, N)
    zeros, ones = torch.zeros(
        ry.size(), dtype=torch.float32).cuda(), torch.ones(
        ry.size(), dtype=torch.float32).cuda()  # (bs, n_p)
    rot_list1 = torch.stack([torch.cos(ry), -torch.sin(ry), zeros], dim=0)
    # (3, bs, np)
    rot_list2 = torch.stack([torch.sin(ry), torch.cos(ry), zeros], dim=0)
    rot_list3 = torch.stack([zeros, zeros, ones], dim=0)
    # (3, bs, n_p)
    rot_list = torch.stack([rot_list1, rot_list2, rot_list3],
                           dim=0)  # (3, 3, bs, N)
    # (3, 3, bs, n_p)

    R_list = rot_list.permute(2, 3, 0, 1)  # (bs, n_p, 3, 3)

    temp_corners = torch.stack([x_corners, y_corners, z_corners],
                               dim=3)  # (bs, n_p, 8, 3)
    rotated_corners = torch.matmul(temp_corners, R_list)  # (bs, n_p, 8, 3)
    x_corners = rotated_corners[:, :, :, 0]  # (bs, n_p, 8, 1)
    y_corners = rotated_corners[:, :, :, 1]
    z_corners = rotated_corners[:, :, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, :, 0], boxes3d[:, :, 1], boxes3d[:, :,
                                                              2]  # (bs, n_p)

    x = torch.unsqueeze(x_loc, 2) + x_corners.reshape(-1, boxes_num,
                                                      8)  # (bs,n_p,8)
    y = torch.unsqueeze(y_loc, 2) + y_corners.reshape(-1, boxes_num, 8)
    z = torch.unsqueeze(z_loc, 2) + z_corners.reshape(-1, boxes_num, 8)

    corners = torch.stack(
        [x, y, z],
        dim=3)  # (bs, n_p, 8, 3)

    return corners.type(torch.float32)  # (bs, n_p, 8, 3)

