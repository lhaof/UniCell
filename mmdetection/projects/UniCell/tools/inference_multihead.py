import os

from mmdet.apis import init_detector
from inference_multi import inference_detector
from mmdet.utils import register_all_modules
import mmcv
import copy
import re
import scipy.io as sio
import numpy as np
from collections import OrderedDict
import scipy
from scipy.optimize import linear_sum_assignment
import time


time_total = 0
image_count = 0

def inference_multihead(config, checkpoint, datasets, patch_dir):
    global time_total, image_count
    register_all_modules()
    model = init_detector(config, checkpoint, device='cuda:0')

    results_dict = OrderedDict()
    paired_all = create_dict(datasets, [])
    unpaired_true_all = create_dict(datasets, [])
    unpaired_pred_all = create_dict(datasets, [])
    true_inst_type_all = create_dict(datasets, [])
    pred_inst_type_all = create_dict(datasets, [])
    true_idx_offset = create_dict(datasets, 0)
    pred_idx_offset = create_dict(datasets, 0)

    def det(pred_centroid, pred_inst_type, img_name, img_idx, dataset_name):
        nonlocal paired_all, unpaired_true_all, unpaired_pred_all, true_inst_type_all, pred_inst_type_all, true_idx_offset, pred_idx_offset

        img_path = os.path.join(whole_ann_file, img_name + ".mat")
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


        distance = 15 if dataset_name == "OCELOT" else 6
        paired, unpaired_true, unpaired_pred = pair_coordinates(
            true_centroid, pred_centroid, distance
        )

        true_idx_offset[dataset_name] = (
            true_idx_offset[dataset_name] + true_inst_type_all[dataset_name][-1].shape[0] if len(
                true_inst_type_all[dataset_name]) != 0 else 0
        )
        pred_idx_offset[dataset_name] = (
            pred_idx_offset[dataset_name] + pred_inst_type_all[dataset_name][-1].shape[0] if len(
                pred_inst_type_all[dataset_name]) != 0 else 0
        )
        true_inst_type_all[dataset_name].append(true_inst_type)
        pred_inst_type_all[dataset_name].append(pred_inst_type)

        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset[dataset_name]
            paired[:, 1] += pred_idx_offset[dataset_name]
            paired_all[dataset_name].append(paired)

        unpaired_true += true_idx_offset[dataset_name]
        unpaired_pred += pred_idx_offset[dataset_name]
        unpaired_true_all[dataset_name].append(unpaired_true)
        unpaired_pred_all[dataset_name].append(unpaired_pred)

    pred_samples = os.listdir(patch_dir)
    # for p in pred_samples:
    #     print(p)
    img_name_oral_last = None
    dataset_name_oral_last = None
    img_idx = 0
    pred_centroid = []
    pred_inst_type = []
    pred_samples.sort()
    for idx, img_patch_name in enumerate(pred_samples):
        img_patch_path = os.path.join(patch_dir, img_patch_name)
        img = mmcv.imread(img_patch_path)
        ret = re.match(r'((.*?)_.*)_(.*)_(.*)_(.*)_(.*).jpg$', img_patch_name)
        img_name_oral = ret.group(1)
        dataset_name_oral = ret.group(2)
        x_start = int(ret.group(3))
        y_start = int(ret.group(4))

        time_new = time.time()
        result = inference_detector(model, img, dataset_id=dataset_id[dataset_name_oral])
        time_total += time.time() - time_new

        image_count += 1

        if idx == 0:
            pred_centroid = []
            pred_inst_type = []
            img_name_oral_last = img_name_oral
            dataset_name_oral_last = dataset_name_oral

        elif x_start == 0 and y_start == 0:
            det(pred_centroid, pred_inst_type, img_name_oral_last, img_idx, dataset_name_oral_last)
            img_name_oral_last = img_name_oral
            dataset_name_oral_last = dataset_name_oral
            # pred info
            pred_centroid = []
            pred_inst_type = []

            img_idx += 1

        # import pdb; pdb.set_trace()
        pred_instances_patch = result.pred_instances.cpu().numpy()
        pos_indx = np.where(pred_instances_patch.scores > 0.5)
        pred_centroid.extend(pred_instances_patch.centroids[pos_indx] + np.array([x_start, y_start]))

        pred_inst_type.extend(pred_instances_patch.labels[pos_indx] + 1)

        if idx == len(pred_samples) - 1:
            det(pred_centroid, pred_inst_type, img_name_oral, img_idx, dataset_name_oral)
            img_name_oral_last = img_name_oral
            dataset_name_oral_last = dataset_name_oral
            # pred info
            pred_centroid = []
            pred_inst_type = []

            img_idx += 1

    results_dict['Overall_F1d'] = 0
    results_dict['Overall_F1c'] = 0

    # import pdb; pdb.set_trace()
    for dataset in datasets:
        local_paired_all = np.concatenate(paired_all[dataset], axis=0) if len(paired_all[dataset]) else np.empty(
            (0, 1))
        local_unpaired_true_all = np.concatenate(unpaired_true_all[dataset], axis=0)
        local_unpaired_pred_all = np.concatenate(unpaired_pred_all[dataset], axis=0)
        local_true_inst_type_all = np.concatenate(true_inst_type_all[dataset], axis=0)
        local_pred_inst_type_all = np.concatenate(pred_inst_type_all[dataset], axis=0)

        local_paired_true_type = local_true_inst_type_all[local_paired_all[:, 0]] if len(paired_all[dataset]) else np.empty((0, 1))
        local_paired_pred_type = local_pred_inst_type_all[local_paired_all[:, 1]] if len(paired_all[dataset]) else np.empty((0, 1))
        local_unpaired_true_type = local_true_inst_type_all[local_unpaired_true_all]
        local_unpaired_pred_type = local_pred_inst_type_all[local_unpaired_pred_all]


        w = [1, 1]
        tp_d = local_paired_pred_type.shape[0]
        fp_d = local_unpaired_pred_type.shape[0]
        fn_d = local_unpaired_true_type.shape[0]

        tp_tn_dt = (local_paired_pred_type == local_paired_true_type).sum()
        fp_fn_dt = (local_paired_pred_type != local_paired_true_type).sum()

        acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)
        f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)

        w = [2, 2, 1, 1]

        type_uid_list = np.unique(local_true_inst_type_all)
        type_uid_list = type_uid_list[type_uid_list != 0]
        type_uid_list = type_uid_list.tolist()

        results_list = [f1_d, acc_type]
        for type_uid in type_uid_list:
            f1_type = _f1_type(
                local_paired_true_type,
                local_paired_pred_type,
                local_unpaired_true_type,
                local_unpaired_pred_type,
                type_uid,
                w,
            )
            results_list.append(f1_type)

        results_dict["F1c_Avg_{}".format(dataset)] = 0
        print("\n============Evaluation {}================".format(dataset))
        print("F1 Detection:{}".format(results_list[0]))
        results_dict["F1d_{}".format(dataset)] = results_list[0]
        results_dict["Overall_F1d"] += results_list[0]
        for i in range(dataset_classes[dataset]):
            print("F1d Type {}:{}".format(
                classes[i + num_classes_offset[dataset]],
                results_list[i + 2]))
            results_dict["F1_Type_{}".format(
                classes[i + num_classes_offset[dataset]])] = results_list[
                i + 2]
            results_dict["F1c_Avg_{}".format(dataset)] += results_list[i + 2]
        results_dict["F1c_Avg_{}".format(dataset)] /= dataset_classes[dataset]
        results_dict["Overall_F1c"] += results_dict["F1c_Avg_{}".format(dataset)]
        print("F1c Avg {}:{}".format(dataset, results_dict["F1c_Avg_{}".format(dataset)]))

    results_dict["Overall_F1d"] /= len(datasets)
    results_dict["Overall_F1c"] /= len(datasets)
    print("\n***************Evaluation Overall******************")
    print("Overall F1 Detection:{}".format(results_dict["Overall_F1d"]))
    print("Overall F1 Classification:{}".format(results_dict["Overall_F1c"]))
    print("***************************************************")
    print("Inference time per image:{}".format(time_total/image_count))


def pair_coordinates(setA, setB, radius):
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

def create_dict(dataset, value):
    return_dict = {}
    for n in dataset:
        return_dict[n] = copy.deepcopy(value)
    return return_dict


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


if __name__ == '__main__':
    path_to_dataset = "Path to Dataset"
    config = "../configs/UniCell_CMOL.py"
    checkpoint = "Path to Checkpoint"
    patch_dir = path_to_dataset + "/Overall_Multi_Patches_FourDataset_CMOL_GIT/Test"
    whole_ann_file = path_to_dataset + "/Overall_Multi_FourDataset_CMOL_GIT/Test_Labels"
    dataset = ["CoNSeP", "MoNuSAC", "OCELOT", "Lizard"]
    dataset_id = {"CoNSeP": 0, "MoNuSAC": 1, "OCELOT": 2, "Lizard": 3}
    num_classes_offset = {"CoNSeP": 0, "MoNuSAC": 3, "OCELOT": 7, "Lizard": 9}
    dataset_classes = {"CoNSeP": 3, "MoNuSAC": 4, "OCELOT": 2, "Lizard": 6}
    classes = ('CoNSeP_Inflammatory', 'CoNSeP_Epithelial', 'CoNSeP_Spindle-shaped',
             'MoNuSAC_Epithelial', 'MoNuSAC_Lymphocyte', 'MoNuSAC_Macrophage', 'MoNuSAC_Neutrophil',
             'OCELOT_Background_Cell', 'OCELOT_Tumor_Cell',
             'Lizard_Neutrophil', "Lizard_Epithelial", "Lizard_Lymphocyte", "Lizard_Plasma",
             "Lizard_Eosinophil", "Lizard_Connective")
    inference_multihead(config, checkpoint, dataset, patch_dir)
