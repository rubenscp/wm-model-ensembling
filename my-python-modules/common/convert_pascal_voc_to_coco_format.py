# ------------------------------------------------------------
# Extract from:
# > How to Convert Annotations from PASCAL VOC XML to COCO JSON
# > https://blog.roboflow.com/how-to-convert-annotations-from-voc-xml-to-coco-json/ 
# ------------------------------------------------------------

import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re
from common.utils import *
from common.manage_log import * 

def get_label2id_list(labels_list: List[str] = None) -> Dict[str, int]:
    """id is 0 start"""
    labels_ids = list(range(0, len(labels_list)+1))
    logging_info(f'get_label2id_list')
    logging_info(f'labels_list: {labels_list}')
    logging_info(f'labels_ids: {labels_ids}')
    diction = dict(zip(labels_list, labels_ids))
    logging_info(f'diction: {diction}')
    return dict(zip(labels_list, labels_ids))

# def get_label2id(labels_path: str) -> Dict[str, int]:
#     """id is 1 start"""
#     with open(labels_path, 'r') as f:
#         labels_str = f.read().split()
#     labels_ids = list(range(1, len(labels_str)+1))
#     return dict(zip(labels_str, labels_ids))

def get_annpaths(annpaths_list_path: str = None) -> List[str]:

    # getting the list of annotation files
    annotation_files = [f for f in os.listdir(annpaths_list_path) if os.path.isfile(os.path.join(annpaths_list_path, f))]

    annotation_files = []
    for file in os.listdir(annpaths_list_path):
        if file.endswith('.xml'):
            annotation_files.append(os.path.join(annpaths_list_path, file))

    # returning the list of annotation files
    return annotation_files

def copy_images_to_folder(input_path: str = None, output_path: str = None):

    # getting the list of image files
    image_files = Utils.get_files_with_extensions(input_path, 'jpg')
    for filename in image_files:
        Utils.copy_file_same_name(filename, input_path, output_path)

# def get_annpaths(ann_dir_path: str = None,
#                  ann_ids_path: str = None,
#                  ext: str = '',
#                  annpaths_list_path: str = None) -> List[str]:
#     """
#     Returns a list of annotation file paths.

#     Args:
#         ann_dir_path (str): The directory path where the annotation files are located.
#         ann_ids_path (str): The path to the file containing the annotation IDs.
#         ext (str): The file extension of the annotation files.
#         annpaths_list_path (str): The path to the file containing a list of annotation file paths.

#     Returns:
#         List[str]: A list of annotation file paths.

#     Raises:
#         FileNotFoundError: If the annotation IDs file or the annotation paths list file is not found.

#     Note:
#         - If `annpaths_list_path` is provided, the function reads the file and returns the annotation paths from it.
#         - If `annpaths_list_path` is not provided, the function reads the annotation IDs from `ann_ids_path` file,
#           appends the `ann_dir_path` and `ext` to each ID, and returns the resulting annotation paths.
#     """

#     # If use annotation paths list
#     if annpaths_list_path is not None:
#         with open(annpaths_list_path, 'r') as f:
#             ann_paths = f.read().split()
#         return ann_paths

#     # If use annotaion ids list
#     ext_with_dot = '.' + ext if ext != '' else ''
#     with open(ann_ids_path, 'r') as f:
#         ann_ids = f.read().split()
#     ann_paths = [os.path.join(ann_dir_path, aid+ext_with_dot) for aid in ann_ids]
#     return ann_paths


def get_image_info(annotation_root, image_filename, image_id,
                   extract_num_from_imgid=True):

    filename, extension = Utils.get_filename_and_extension(image_filename)
    image_filename = filename + '.jpg'

    # path = annotation_root.findtext('path')
    # if path is None:
    #     filename = annotation_root.findtext('filename')
    # else:
    #     filename = os.path.basename(path)
    # img_name = os.path.basename(filename)
    # img_id = os.path.splitext(img_name)[0]
    # if extract_num_from_imgid and isinstance(img_id, str):
    #     img_id = int(re.findall(r'\d+', img_id)[0])
    img_id = image_id

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': image_filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info

def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'area': o_width * o_height,
        'iscrowd': 0,
        'bbox': [xmin, ymin, o_width, o_height],
        'category_id': category_id,
        'ignore': 0,
        'segmentation': []  # This script is not for segmentation
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_folder: str,
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True, 
                             ):
    output_json_dict = {
        "images": [],
        "type": "instances",
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    image_id = 1  # START_IMAGE_ID, TODO input as args ?
    print(f'Start converting folder to COCO format json file: {output_jsonpath}')
    for a_path in tqdm(annotation_paths):

        path, filename_with_extension, filename, extension = Utils.get_filename(a_path)       
        image_filename = os.path.join(output_folder, filename + '.jpg')

        print(f'path: {path}')
        print(f'filename_with_extension: {filename_with_extension}')
        print(f'filename: {filename}')
        print(f'extension: {extension}')
        print(f'image_filename: {image_filename}')

        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  image_filename=image_filename,
                                  image_id=image_id,
                                  extract_num_from_imgid=extract_num_from_imgid
                                  )
        # img_id = img_info['id']
        img_id = image_id
        image_id += 1
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)   

    with open(output_jsonpath, 'w') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


# def main():
#     parser = argparse.ArgumentParser(
#         description='This script support converting voc format xmls to coco format json')
#     parser.add_argument('--ann_dir', type=str, default=None,
#                         help='path to annotation files directory. It is not need when use --ann_paths_list')
#     parser.add_argument('--ann_ids', type=str, default=None,
#                         help='path to annotation files ids list. It is not need when use --ann_paths_list')
#     parser.add_argument('--ann_paths_list', type=str, default=None,
#                         help='path of annotation paths list. It is not need when use --ann_dir and --ann_ids')
#     parser.add_argument('--labels', type=str, default=None,
#                         help='path to label list.')
#     parser.add_argument('--output', type=str, default='output.json', help='path to output json file')
#     parser.add_argument('--ext', type=str, default='', help='additional extension of annotation file')
#     args = parser.parse_args()

#     label2id = get_label2id(labels_path=args.labels)
#     ann_paths = get_annpaths(
#         ann_dir_path=args.ann_dir,
#         ann_ids_path=args.ann_ids,
#         ext=args.ext,
#         annpaths_list_path=args.ann_paths_list
#     )
#     convert_xmls_to_cocojson(
#         annotation_paths=ann_paths,
#         label2id=label2id,
#         output_jsonpath=args.output,
#         extract_num_from_imgid=True
#     )


# if __name__ == '__main__':
#     main()