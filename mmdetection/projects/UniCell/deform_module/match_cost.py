from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmdet.registry import TASK_UTILS
import torch

@TASK_UTILS.register_module()
class NucleiL1Cost(BaseMatchCost):
    def __init__(self, weight):
        super().__init__(weight=weight)

    def __call__(self, pred_instances, gt_instances, img_meta, **kwargs):
        pred_points = pred_instances.centroids
        gt_points = gt_instances.centroids

        img_h, img_w = img_meta['img_shape']
        factor = gt_points.new_tensor([img_w, img_h]).unsqueeze(0)

        gt_points = gt_points / factor
        pred_points = pred_points / factor

        point_cost = torch.cdist(pred_points, gt_points, p=1)
        return point_cost * self.weight