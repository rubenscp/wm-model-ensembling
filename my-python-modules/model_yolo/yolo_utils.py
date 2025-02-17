"""
Project: White Mold 
Description: Utils methods and functions 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 04/04/2024
Version: 1.0
"""
# Importing Python libraries 
import os
import shutil
import json 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Importing python modules
from common.manage_log import *
from common.utils import * 

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class YoloUtils:

    # Copy all image files from folder list to one specific folder 
    @staticmethod
    def merge_image_results_to_one_folder(results_folder, folder_prefix, 
                                          test_image_folder, test_image_sufix):

        # getting list of folder based on folder prefix
        source_folders = Utils.get_folders(results_folder)
        logging_info(f'source_folders: {source_folders}')

        # copying all files in each source folder
        for source_folder in source_folders:
            # checking if folder name contains the same folder prefix 
            if folder_prefix in source_folder:
                input_folder = os.path.join(results_folder, source_folder)
                image_filenames = Utils.get_files_with_extensions(input_folder, 'jpg')

                # copying image files 
                for image_filename in image_filenames:
                    filename, extension = Utils.get_filename_and_extension(image_filename)
                    teste_image_filename = filename + test_image_sufix + '.' + extension 
                    Utils.copy_file(image_filename, input_folder, 
                                    teste_image_filename, test_image_folder)
                    
