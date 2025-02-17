
from common.utils import *

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'
NEW_FILE = True

def create_project_yaml_file_for_train_valid(
    path_and_filename_white_mold_yaml,
    image_dataset_folder,
    classes):

    # removing yaml file, if exists 
    Utils.remove_file(path_and_filename_white_mold_yaml)
    
    # building text to write into the file     
    lines = ''
    lines += 'White Mold Yaml Setup:' + LINE_FEED + LINE_FEED
    lines += 'path: ' + image_dataset_folder + LINE_FEED + LINE_FEED
    lines += 'train: train/images' + LINE_FEED
    lines += 'val: valid/images' + LINE_FEED
    lines += 'test: test/images' + LINE_FEED
    lines += LINE_FEED
    lines += '# class names' + LINE_FEED
    lines += 'nc: ' + str(len(classes)) + LINE_FEED
    lines += 'names: ' + str(classes) + LINE_FEED

    # writing file 
    Utils.save_text_file(path_and_filename_white_mold_yaml, lines, NEW_FILE)

def create_project_yaml_file_for_test(
    path_and_filename_white_mold_yaml,
    image_dataset_folder,
    classes):

    # removing yaml file, if exists 
    Utils.remove_file(path_and_filename_white_mold_yaml)
    
    # building text to write into the file     
    lines = ''
    lines += 'White Mold Yaml Setup:' + LINE_FEED + LINE_FEED
    lines += 'path: ' + image_dataset_folder + LINE_FEED + LINE_FEED
    lines += 'train: train/images' + LINE_FEED
    lines += 'val: test/images' + LINE_FEED
    lines += 'test: test/images' + LINE_FEED
    lines += LINE_FEED
    lines += '# class names' + LINE_FEED
    lines += 'nc: ' + str(len(classes)) + LINE_FEED
    lines += 'names: ' + str(classes) + LINE_FEED

    # writing file 
    Utils.save_text_file(path_and_filename_white_mold_yaml, lines, NEW_FILE)    