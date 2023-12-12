'''
Category for dataset
CoNSeP: Inflammatory, Epithelial, Stromal
BRCA-M2C: Inflammatory, Epithelial, Stromal
PanNuke: Neoplastic, Epithelial, Inflammatory, Connective, Dead
Lizard: Epithelial, Lymphocytes, Plasma, Neutrophils, Eosinophil, Connective
MoNuSAC: Epithelial, Lymphocytes, Macrophages, Neutrophils
OCELOT: 'Background Cell', 'Tumor Cell'
'''

import os
import pandas as pd
import scipy.io as sio
import mmcv
import numpy as np
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation, create_coco_dict
from sahi.utils.file import save_json
from pathlib import Path
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def trans_to_coco_multi_cate(dataset_path, mode, img_save_path, Save_Centroids=True):
    categories = ['CoNSeP_Inflammatory', 'CoNSeP_Epithelial', 'CoNSeP_Spindle-shaped',
                  'MoNuSAC_Epithelial', 'MoNuSAC_Lymphocyte', 'MoNuSAC_Macrophage', 'MoNuSAC_Neutrophil',
                  'OCELOT_Background_Cell', 'OCELOT_Tumor_Cell',
                  'Lizard_Neutrophil', "Lizard_Epithelial", "Lizard_Lymphocyte", "Lizard_Plasma",
                  "Lizard_Eosinophil", "Lizard_Connective"]
    coco = Coco()
    for idx, cat in enumerate(categories):
        coco.add_category(CocoCategory(id=idx + 1, name=categories[idx]))

    os.makedirs(img_save_path, exist_ok=True)
    os.makedirs(os.path.join(img_save_path, mode), exist_ok=True)
    os.makedirs(os.path.join(img_save_path, mode + "_Labels"), exist_ok=True)


    datasets = ["CoNSeP", "MoNuSAC", "OCELOT", "Lizard"]
    classes_offset = [0, 3, 7, 9]

    for data_idx, dataset in enumerate(datasets):
        imgs_path = os.path.join(dataset_path, dataset, mode, "Images")
        anns_path = os.path.join(dataset_path, dataset, mode, "Labels")
        imgs_files = os.listdir(imgs_path)

        for imgid, image in enumerate(imgs_files):
            repeat_num = 1
            image_save_name = dataset + "_" + image
            ann_save_name = dataset + "_" + image[:-4] + '.mat'
            print("Processing ", image_save_name)
            img = mmcv.imread(os.path.join(imgs_path, image))
            if dataset == "MoNuSAC" and mode == "Train":
                if img.shape[0] <= 320 or img.shape[1] <= 320:
                    repeat_num = 4

            for rep in range(repeat_num):
                if mode == "Train":
                    image_save_name = dataset + "_" + image[:-4] + "_" + str(rep) + '.png'
                    ann_save_name = dataset + "_" + image[:-4] + "_" + str(rep) + '.mat'
                coco_image = CocoImage(file_name=image_save_name, height=img.shape[0], width=img.shape[1])

                classes_ann = sio.loadmat(os.path.join(anns_path, image[:-4] + '.mat'))

                if dataset == 'CoNSeP':
                    true_inst_type = classes_ann["inst_type"].copy()
                    true_inst_type[(true_inst_type == 2)] = 9
                    true_inst_type[(true_inst_type == 3) | (true_inst_type == 4)] = 2
                    true_inst_type[
                        (true_inst_type == 1) | (true_inst_type == 5) | (true_inst_type == 6) | (true_inst_type == 7)] = 3
                    true_inst_type[(true_inst_type == 9)] = 1
                    classes_ann["inst_type"] = true_inst_type

                try:
                    if not len(classes_ann["inst_type"]) == len(classes_ann["inst_centroid"]):
                        continue
                except:
                    pass
                # assert len(classes_ann["inst_type"]) == max(np.unique(classes_ann["inst_map"])), print(
                #     "Vertify, Something wrong happen")

                # Saveing Centroids Only:
                if Save_Centroids == True:
                    for inst_num in range(len(classes_ann["inst_type"])):
                        centroid = classes_ann["inst_centroid"][inst_num]
                        cate = int(classes_ann["inst_type"][inst_num][0]) + classes_offset[data_idx]
                        if len(CocoAnnotation(
                                bbox=[centroid[0] - 2.0, centroid[1] - 2.0, 4.0, 4.0],
                                category_id=cate,
                                category_name=categories[cate - 1]
                        ).bbox) == 0:
                            print(111111111111111)
                            continue
                        coco_image.add_annotation(
                            CocoAnnotation(
                                bbox=[centroid[0] - 2.0, centroid[1] - 2.0, 4.0, 4.0],
                                category_id=cate,
                                category_name=categories[cate - 1]
                            )
                        )

                coco.add_image(coco_image)

                mmcv.imwrite(img, os.path.join(img_save_path, mode, image_save_name))
                sio.savemat(os.path.join(img_save_path, mode + "_Labels", ann_save_name), classes_ann)


    save_json(data=coco.json, save_path=os.path.join(img_save_path, "annotations", "COCO_{}_multiabsolute_{}.json".format(mode,
                                                                                                                   "Centroids" if Save_Centroids else "Segmentation")))


def trans_to_patch(json_original_path, img_orinial_path, json_save_path, patch_save_path, mode):
    from sahi.slicing import slice_coco, slice_image
    from sahi.utils.file import load_json
    coco_dict = load_json(json_original_path)
    coco = Coco.from_coco_dict_or_path(coco_dict)
    sliced_coco_images = []

    coco_api = COCO(json_original_path)

    for coco_image in coco.images:
        print("Slicing :", coco_image.file_name)

        image_path = os.path.join(img_orinial_path, coco_image.file_name)
        if coco_image.file_name.startswith('OCELOT'):
            patch_size = 512
        elif coco_image.file_name.startswith('MoNuSAC'):
            patch_size = 320
        else:
            patch_size = 250
        patch_scale_height = coco_image.height // patch_size
        patch_scale_width = coco_image.width // patch_size
        if coco_image.file_name.startswith('BRCA'):
            slice_height = coco_image.height // patch_scale_height if patch_scale_height != 1 else coco_image.height // 2
            slice_width = coco_image.width // patch_scale_width if patch_scale_width != 1 else coco_image.width // 2
        else:
            slice_height = coco_image.height // patch_scale_height if patch_scale_height != 0 else coco_image.height
            slice_width = coco_image.width // patch_scale_width if patch_scale_width != 0 else coco_image.width

        if mode == "Train":
            overlap_width_ratio = 0.2
            overlap_height_ratio = 0.2
            if coco_image.file_name.startswith('CoNSeP'):
                overlap_width_ratio = 0.8
                overlap_height_ratio = 0.8
            elif coco_image.file_name.startswith('BRCA'):
                overlap_width_ratio = 0.5
                overlap_height_ratio = 0.5
            elif coco_image.file_name.startswith('OCELOT'):
                overlap_width_ratio = 0.0
                overlap_height_ratio = 0.0
            elif coco_image.file_name.startswith('MoNuSAC'):
                overlap_width_ratio = 0.4
                overlap_height_ratio = 0.4
        else:
            overlap_width_ratio = 0.0
            overlap_height_ratio = 0.0


        slice_image_result = slice_image(
            image=image_path,
            coco_annotation_list=coco_image.annotations,
            output_file_name=Path(coco_image.file_name).stem,
            output_dir=patch_save_path,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            min_area_ratio=0.1,
            out_ext=None,
            verbose=True,
            mode=mode,
        )
        sliced_coco_images.extend(slice_image_result.coco_images)

    coco_dict = create_coco_dict(
        sliced_coco_images, coco_dict["categories"], ignore_negative_samples=False
    )

    save_json(coco_dict, json_save_path)


def valid(save):
    coco = COCO(save)

    # 统计类别
    cats = coco.loadCats(coco.getCatIds())
    cat_nms = [cat['name'] for cat in cats]
    print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))
    for cat_name in cat_nms:
        catId = coco.getCatIds(catNms=cat_name)
        imgId = coco.getImgIds(catIds=catId)
        annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)

        print("{:<15} {:<6d}     {:<10d}".format(cat_name, len(imgId), len(annId)))

    return


if __name__ == '__main__':
    save_dir = "Overall_Multi_FourDataset_CMOL_GIT"
    patch_save_dir = "Overall_Multi_Patches_FourDataset_CMOL_GIT"
    dataset_path = "/mntnfs/med_data2/huangjunjia/dataset/PromptDet/Dataset"
    img_save_path = "{}/{}".format(dataset_path, save_dir)
    modes = ["Train", "Test"]
    Save_Centroids = True
    for mode in modes:
        trans_to_coco_multi_cate(dataset_path, mode, img_save_path, Save_Centroids=Save_Centroids)
        trans_to_patch(
            json_original_path="{}/{}/annotations/COCO_{}_multiabsolute_Centroids.json".format(dataset_path, save_dir,
                mode),
            json_save_path="{}/{}/annotations/COCO_{}_multiabsolute_Centroids_SAHI.json".format(dataset_path,
                patch_save_dir, mode),
            img_orinial_path="{}/{}/{}".format(dataset_path, save_dir, mode),
            patch_save_path="{}/{}/{}".format(dataset_path, patch_save_dir, mode),
            mode=mode)
        valid(
            "{}/{}/annotations/COCO_{}_multiabsolute_Centroids.json".format(dataset_path, save_dir,
                mode))
