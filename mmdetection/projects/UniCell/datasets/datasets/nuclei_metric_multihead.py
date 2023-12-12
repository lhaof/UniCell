# Copyright (c) OpenMMLab. All rights reserved.
import copy
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import FileClient, dump, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from mmdet.evaluation.functional import eval_recalls
import re
import os
import scipy.io as sio
import scipy
from scipy.optimize import linear_sum_assignment


@METRICS.register_module()
class NucleiMetric_MultiHead(BaseMetric):
    default_prefix: Optional[str] = None

    def __init__(self,
                 ann_file: Optional[str] = None,
                 whole_ann_file: Optional[str] = None,
                 dataset_name: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = dict(backend='disk'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.whole_ann_file = whole_ann_file
        self.dataset = dataset_name
        allowed_metrics = ['centroids', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'centroids', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.file_client_args = file_client_args
        self.file_client = FileClient(**file_client_args)

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with self.file_client.get_local_path(ann_file) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = self._coco_api.get_cat_ids()
        self.img_ids = None

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['centroids'] = pred['centroids'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                    pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            self.results.append(result)

    def det(self, pred_centroid, pred_inst_type, img_name, img_idx, dataset_name):
        img_path = os.path.join(self.whole_ann_file, img_name + ".mat")
        true_info = sio.loadmat(img_path)
        true_centroid = (true_info["inst_centroid"]).astype("float32")
        true_inst_type = (true_info["inst_type"]).astype("int32")

        if true_centroid.shape[0] != 0:
            true_inst_type = true_inst_type[:, 0]
        else:  # no instance at all
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        pred_centroid = np.asarray(pred_centroid).astype("float32")
        pred_inst_type = np.asarray(pred_inst_type).astype("int32")

        if pred_centroid.shape[0] != 0:
            pred_inst_type = pred_inst_type[:]
        else:  # no instance at all
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # assert self.dataset == 'Overall_MultiHead', 'Only support Overall_MultiHead'
        # if img_name.startswith("CoNSeP"):
        #     true_inst_type[(true_inst_type == 2)] = 9
        #     true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 2
        #     true_inst_type[(true_inst_type == 1)] = 3
        #     true_inst_type[
        #         (true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 3
        #     true_inst_type[(true_inst_type == 9)] = 1

        if self.dataset == "Overall_Binary":
            true_inst_type[(true_inst_type > 0)] = 1

        distance = 15 if dataset_name == "OCELOT" else 6
        paired, unpaired_true, unpaired_pred = self.pair_coordinates(
            true_centroid, pred_centroid, distance
        )

        self.true_idx_offset[dataset_name] = (
            self.true_idx_offset[dataset_name] + self.true_inst_type_all[dataset_name][-1].shape[0] if len(self.true_inst_type_all[dataset_name]) != 0 else 0
        )
        self.pred_idx_offset[dataset_name] = (
            self.pred_idx_offset[dataset_name] + self.pred_inst_type_all[dataset_name][-1].shape[0] if len(self.pred_inst_type_all[dataset_name]) != 0 else 0
        )
        self.true_inst_type_all[dataset_name].append(true_inst_type)
        self.pred_inst_type_all[dataset_name].append(pred_inst_type)

        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += self.true_idx_offset[dataset_name]
            paired[:, 1] += self.pred_idx_offset[dataset_name]
            self.paired_all[dataset_name].append(paired)

        unpaired_true += self.true_idx_offset[dataset_name]
        unpaired_pred += self.pred_idx_offset[dataset_name]
        self.unpaired_true_all[dataset_name].append(unpaired_true)
        self.unpaired_pred_all[dataset_name].append(unpaired_pred)

    def create_dict(self, dataset, value):
        return_dict = {}
        for n in dataset:
            return_dict[n] = copy.deepcopy(value)
        return return_dict

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        results_dict = OrderedDict()
        # split gt and prediction list
        preds = results
        logger.info(f'Evaluating {self.metrics[0]}...')

        self.paired_all = self.create_dict(self.dataset_meta['datasets'], [])
        self.unpaired_true_all = self.create_dict(self.dataset_meta['datasets'], [])
        self.unpaired_pred_all = self.create_dict(self.dataset_meta['datasets'], [])
        self.true_inst_type_all = self.create_dict(self.dataset_meta['datasets'], [])
        self.pred_inst_type_all = self.create_dict(self.dataset_meta['datasets'], [])
        self.true_idx_offset = self.create_dict(self.dataset_meta['datasets'], 0)
        self.pred_idx_offset = self.create_dict(self.dataset_meta['datasets'], 0)


        img_name_oral_last = None
        dataset_name_oral_last = None
        img_idx = 0
        pred_centroid = []
        pred_inst_type = []
        for idx, pred in enumerate(preds):
            img_id = pred['img_id']
            img_info = self._coco_api.load_imgs(img_id)
            img_name = img_info[0]['file_name']
            ret = re.match(r'((.*?)_.*)_(.*)_(.*)_(.*)_(.*).jpg$', img_name)
            img_name_oral = ret.group(1)
            dataset_name_oral = ret.group(2)
            x_start = int(ret.group(3))
            y_start = int(ret.group(4))

            if idx == 0:
                pred_centroid = []
                pred_inst_type = []
                img_name_oral_last = img_name_oral
                dataset_name_oral_last = dataset_name_oral

            elif x_start == 0 and y_start == 0:
                self.det(pred_centroid, pred_inst_type, img_name_oral_last, img_idx, dataset_name_oral_last)
                img_name_oral_last = img_name_oral
                dataset_name_oral_last = dataset_name_oral
                # pred info
                pred_centroid = []
                pred_inst_type = []

                img_idx += 1

            score = 0.5
            pos_indx = np.where(pred['scores'] > score)
            pred_centroid.extend(pred['centroids'][pos_indx] + np.array([x_start, y_start]))
            pred_inst_type.extend(pred['labels'][pos_indx] + 1)

            if idx == len(results) - 1:
                self.det(pred_centroid, pred_inst_type, img_name_oral, img_idx, dataset_name_oral)
                img_name_oral_last = img_name_oral
                dataset_name_oral_last = dataset_name_oral
                # pred info
                pred_centroid = []
                pred_inst_type = []

                img_idx += 1

        results_dict['Overall_F1d'] = 0
        results_dict['Overall_F1c'] = 0
        for dataset in self.dataset_meta['datasets']:
            paired_all = np.concatenate(self.paired_all[dataset], axis=0) if len(self.paired_all[dataset]) else np.empty((0, 1))
            unpaired_true_all = np.concatenate(self.unpaired_true_all[dataset], axis=0)
            unpaired_pred_all = np.concatenate(self.unpaired_pred_all[dataset], axis=0)
            true_inst_type_all = np.concatenate(self.true_inst_type_all[dataset], axis=0)
            pred_inst_type_all = np.concatenate(self.pred_inst_type_all[dataset], axis=0)

            paired_true_type = true_inst_type_all[paired_all[:, 0]] if len(self.paired_all[dataset]) else np.empty((0, 1))
            paired_pred_type = pred_inst_type_all[paired_all[:, 1]] if len(self.paired_all[dataset]) else np.empty((0, 1))
            unpaired_true_type = true_inst_type_all[unpaired_true_all]
            unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

            def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
                type_samples = (paired_true == type_id) | (paired_pred == type_id)

                paired_true = paired_true[type_samples]
                paired_pred = paired_pred[type_samples]

                tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
                tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
                fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
                fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

                fp_d = (unpaired_pred == type_id).sum()
                fn_d = (unpaired_true == type_id).sum()

                f1_type = (2 * (tp_dt + tn_dt)) / (
                        2 * (tp_dt + tn_dt)
                        + w[0] * fp_dt
                        + w[1] * fn_dt
                        + w[2] * fp_d
                        + w[3] * fn_d
                )
                return f1_type

            w = [1, 1]
            tp_d = paired_pred_type.shape[0]
            fp_d = unpaired_pred_type.shape[0]
            fn_d = unpaired_true_type.shape[0]

            tp_tn_dt = (paired_pred_type == paired_true_type).sum()
            fp_fn_dt = (paired_pred_type != paired_true_type).sum()

            acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
            f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

            w = [2, 2, 1, 1]

            # if type_uid_list is None:
            type_uid_list = np.unique(true_inst_type_all)
            type_uid_list = type_uid_list[type_uid_list != 0]
            type_uid_list = type_uid_list.tolist()

            results_list = [f1_d, acc_type]
            for type_uid in type_uid_list:
                f1_type = _f1_type(
                    paired_true_type,
                    paired_pred_type,
                    unpaired_true_type,
                    unpaired_pred_type,
                    type_uid,
                    w,
                )
                results_list.append(f1_type)

            # results_dict = {}
            results_dict["F1c_Avg_{}".format(dataset)] = 0
            logger.info("\n============Evaluation {}================".format(dataset))
            logger.info("F1 Detection:{}".format(results_list[0]))
            results_dict["F1d_{}".format(dataset)] = results_list[0]
            results_dict["Overall_F1d"] += results_list[0]
            for i in range(self.dataset_meta['dataset_classes'][dataset]):
                logger.info("F1d Type {}:{}".format(self.dataset_meta['classes'][i+self.dataset_meta['num_classes_offset'][dataset]], results_list[i + 2]))
                results_dict["F1_Type_{}".format(self.dataset_meta['classes'][i+self.dataset_meta['num_classes_offset'][dataset]])] = results_list[i + 2]
                results_dict["F1c_Avg_{}".format(dataset)] += results_list[i + 2]
            results_dict["F1c_Avg_{}".format(dataset)] /= self.dataset_meta['dataset_classes'][dataset]
            results_dict["Overall_F1c"] += results_dict["F1c_Avg_{}".format(dataset)]
            logger.info("F1c Avg {}:{}".format(dataset, results_dict["F1c_Avg_{}".format(dataset)]))

        results_dict["Overall_F1d"] /= len(self.dataset_meta['datasets'])
        results_dict["Overall_F1c"] /= len(self.dataset_meta['datasets'])
        logger.info("\n***************Evaluation Overall******************")
        logger.info("Overall F1 Detection:{}".format(results_dict["Overall_F1d"]))
        logger.info("Overall F1 Classification:{}".format(results_dict["Overall_F1c"]))
        return results_dict

    def pair_coordinates(self, setA, setB, radius):
        """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
        unique pairing (largest possible match) when pairing points in set B
        against points in set A, using distance as cost function.

        Args:
            setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                        of N different points
            radius: valid area around a point in setA to consider
                    a given coordinate in setB a candidate for match
        Return:
            pairing: pairing is an array of indices
            where point at index pairing[0] in set A paired with point
            in set B at index pairing[1]
            unparedA, unpairedB: remaining poitn in set A and set B unpaired

        """
        # * Euclidean distance as the cost matrix
        pair_distance = scipy.spatial.distance.cdist(setA, setB, metric='euclidean')

        # * Munkres pairing with scipy library
        # the algorithm return (row indices, matched column indices)
        # if there is multiple same cost in a row, index of first occurence
        # is return, thus the unique pairing is ensured
        indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

        # extract the paired cost and remove instances
        # outside of designated radius
        pair_cost = pair_distance[indicesA, paired_indicesB]

        pairedA = indicesA[pair_cost <= radius]
        pairedB = paired_indicesB[pair_cost <= radius]

        pairing = np.concatenate([pairedA[:, None], pairedB[:, None]], axis=-1)
        unpairedA = np.delete(np.arange(setA.shape[0]), pairedA)
        unpairedB = np.delete(np.arange(setB.shape[0]), pairedB)
        return pairing, unpairedA, unpairedB
