"""
Project: White Mold 
Description: Utils methods and functions 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 20/10/2023
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

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class Utils:

    # Create a folder
    @staticmethod
    def create_directory(folder):
        if not os.path.isdir(folder):    
            os.makedirs(folder)

    # Remove all files from a folder
    @staticmethod
    def remove_directory(folder):
        shutil.rmtree(folder, ignore_errors=True)

    # Read list of datasets in the input folder
    @staticmethod
    def get_folders(input_path):

        # getting list of image folders 
        folders = [f for f in os.listdir(input_path) 
                              if os.path.isdir(os.path.join(input_path, f))]        
        
        # returning list of image folders
        return folders

    # Read list of supervisely annotation files of one specific folder 
    def get_files_with_extensions(folder, extension):

        # getting list of supervisely annotation files
        files = [f for f in os.listdir(folder) if f.endswith(extension)]
        
        # returning list of image folders
        return files

    # Copy one file
    @staticmethod
    def copy_file_same_name(filename, input_path, output_path):
        source = os.path.join(input_path, filename)
        destination = os.path.join(output_path, filename)
        shutil.copy(source, destination)
        # copy_file(input_filename=filename, input_path=input_path, output_filename=filename, output_path=output_path)

    @staticmethod
    def copy_file(input_filename, input_path, output_filename, output_path):
        source = os.path.join(input_path, input_filename)
        destination = os.path.join(output_path, output_filename)
        shutil.copy(source, destination)

    # Remove one file
    @staticmethod
    def remove_file(path_and_filename):
        if os.path.isfile(path_and_filename): 
            os.remove(path_and_filename) 

    # # Read image 
    # @staticmethod
    # def read_image(filename, path):
    #     path_and_filename = os.path.join(path, filename)
    #     image = cv2.imread(path_and_filename)
    #     return image

    # # Save image
    # @staticmethod
    # def save_image(path_and_filename, image):
    #     cv2.imwrite(path_and_filename, image)

    # Save text file 
    @staticmethod
    def save_text_file(path_and_filename, content_of_text_file, create):
        # setting annotation file
        text_file = open(path_and_filename, 'w' if create else 'a+')

        # write line
        text_file.write(content_of_text_file)

        # closing text file
        text_file.close()        

    # Read text file 
    @staticmethod
    def read_text_file(path_and_filename):
        # setting annotation file
        text_file = open(path_and_filename, 'r')

        # reading the file 
        lines = text_file.readlines()

        # replacing end of line('/n') with ''
        data_into_list = [line.replace('\n','') for line in lines]
        
        # closing text file
        text_file.close()        

        # returning text in list format 
        return data_into_list


    # # Draw bounding box in the image
    # @staticmethod
    # def draw_bounding_box(image, linP1, colP1, linP2, colP2, bgrBoxColor, thickness, label):
    #     # Start coordinate represents the top left corner of rectangle
    #     startPoint = (colP1, linP1)

    #     # Ending coordinate represents the bottom right corner of rectangle
    #     endPoint = (colP2, linP2)

    #     # Draw a rectangle with blue line borders of thickness of 2 px
    #     image = cv2.rectangle(image, startPoint, endPoint, bgrBoxColor, thickness)

    #     # setting the bounding box label
    #     cv2.putText(image, label,
    #                 (colP1, linP1 - 5),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgrBoxColor, 2)


    #     # returning the image with bounding box drawn
    #     return image

    # Create zip file from a directory 
    @staticmethod
    def zip_directory(source_directory, output_filename):
        shutil.make_archive(output_filename, 'zip', source_directory)

        # Check to see if the zip file is created
        full_output_filename = output_filename + '.zip'
        if os.path.exists(full_output_filename):
            return True, full_output_filename
        else:
            return False, None
        

    # Read JSON file with execution parameters
    @staticmethod
    def read_json_parameters(filename):

        # defining  parameters dictionary
        parameters = {}

        # reading parameters file 
        with open(filename) as json_file:
            parameters = json.load(json_file)

        # returning  parameters dictionary
        return parameters

    # Convert json boolean to python boolean 
    def to_boolean_value(json_boolean_value):
        boolean_value = bool(json_boolean_value == 'true')

    # Create a pretty json for printing
    @staticmethod
    def get_pretty_json(json_text):

        # formatting pretty json
        json_formatted_str = json.dumps(json_text, indent=4)
        
        # returning json formatted
        return json_formatted_str

    # Get filename from full string 
    @staticmethod
    def get_filename(path_and_filename):
        # getting filename with extension
        index_1 = path_and_filename.rfind('/')
        path = path_and_filename[:index_1]
        filename_with_extension = path_and_filename[index_1 + 1:]

        # getting just the filename and the extension separated
        index_2 = path_and_filename.rfind('.')
        filename = path_and_filename[index_1 + 1:index_2]
        extension = path_and_filename[index_2 + 1:]

        # print(f'path_and_filename:       {path_and_filename}')
        # print(f'path:                    {path}')
        # print(f'filename_with_extension: {filename_with_extension}')
        # print(f'filename:                {filename}')
        # print(f'extension:               {extension}')

        # returning path, filename with extension, just filename and the extension 
        return path, filename_with_extension, filename, extension
                        
    # Get filename from full string 
    @staticmethod
    def get_filename_and_extension(filename_with_extension):    
        # getting just the filename and the extension separated
        index = filename_with_extension.rfind('.')
        filename = filename_with_extension[:index]
        extension = filename_with_extension[index + 1:]

        # returning filename with extension
        return filename, extension                    

    # Extract from 
    # https://stackoverflow.com/questions/20927368/how-to-normalize-a-confusion-matrix
    @staticmethod
    def save_plot_confusion_matrix(confusion_matrix, path_and_filename, title,
                                   format, x_labels_names, y_labels_names):
        
        # logging_info(f'Plotting confusion matrix normalized')
        # logging_info(f'classes: {classes}')

        # x_labels_names = classes.copy()
        # y_labels_names = classes.copy()
        # x_labels_names.append('Ghost predictions')    
        # y_labels_names.append('Undetected objects')

        fig, ax = plt.subplots(figsize=(14,10))
        heatmap_axis = sns.heatmap(
            confusion_matrix, 
            annot=True, fmt=format, 
            xticklabels=x_labels_names, 
            yticklabels=y_labels_names,
            linewidth=0.5,
            cmap="crest",
            annot_kws={'size': 11}
        )
        plt.title(title + LINE_FEED)
        plt.xlabel('Actual (Ground Truth)')
        plt.ylabel('Predicted (By the model)')
        plt.show(block=False)
        heatmap_axis.xaxis.tick_bottom()
        fig.savefig(path_and_filename)

    @staticmethod
    def save_plot(values, path_and_filename, title, x_label, y_label):       
        fig = plt.figure(figsize=(10, 7), num=1, clear=True)
        ax = fig.add_subplot()
        ax.plot(values, color='tab:blue')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.title(title + LINE_FEED)
        # do not print point values 
        # for i, value in enumerate(values):
        #     plt.annotate('%.2f' % value, xy=(i, value), size=8)
        fig.savefig(path_and_filename)

    @staticmethod
    def save_losses(losses, path_and_filename):
        ''' 
        Save losses values into MSExcel file
        '''

        # preparing columns name to list
        column_names = [
            'epoch',
            'loss',
        ]

        # creating dataframe from list 
        df = pd.DataFrame(losses, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='losses', index=False)

    @staticmethod
    def save_losses_maps(losses, maps, maps_50, maps_75, path_and_filename):
        ''' 
        Save losses values into MSExcel file
        '''

        # preparing columns name to list
        column_names = [
            'epoch',
            'loss',
            'map50-95', 
            'map50',
            'map75',
        ]

        # preparing list
        list = []
        for i in range(len(losses)):
            item = [(i+1), losses[i], maps[i], maps_50[i], maps_75[i] ]
            list.append(item)

        # creating dataframe from list 
        df = pd.DataFrame(list, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='loss_and_map', index=False)

    @staticmethod
    def save_confusion_matrix_excel(
        confusion_matrix, path_and_filename, x_labels_names, y_labels_names, 
        tp_per_classes, fp_per_classes, fn_per_classes, tn_per_classes):

        # preparing columns name to list
        column_names = np.hstack(('', x_labels_names))
        # column_names.append('TP')
        # column_names.append('FP')
        # column_names.append('FN')
        # column_names.append('TN')
        print(f'column_names: {column_names}')
        print(f'confusion_matrix: {confusion_matrix}')
        print(f' tp_per_classes: { tp_per_classes}')

        # building sheet
        list = []
        i = 0
        for item in confusion_matrix:
            print(f'item: {item}')
            row = np.hstack((y_labels_names[i], item))
            list.append(row.tolist())
            i += 1

        list.append('')            
        list.append(['', 'TP', 'FP', 'FN', 'TN'])
        print(f'rubens')
        print(f'len(tp_per_classes): {len(tp_per_classes)}')
        print(f'fp_per_classes: {fp_per_classes}')
        print(f'fn_per_classes: {fn_per_classes}')
        print(f'tp_per_classes: {tp_per_classes}')
        print(f'x_labels_names: {x_labels_names}')
        if x_labels_names[0] == '__background__':
            classes = x_labels_names[1:]
        else:           
            classes = x_labels_names
        print(f'classes: {classes}')

        for i in range(len(tp_per_classes)):
            print(f'i: {i}')
            row = [classes[i], tp_per_classes[i], fp_per_classes[i], fn_per_classes[i], tn_per_classes[i]]
            list.append(row)

        # print(f'column_names: {column_names}')
        print(f'list: {list}')

        # creating dataframe from list 
        df = pd.DataFrame(list, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='confusion_matrix', index=False)

    @staticmethod
    def save_metrics_excel(path_and_filename, sheet_name,sheet_list):

        # preparing columns name to list
        # column_names = ['A', 'B']

        # creating dataframe from list        
        # df = pd.DataFrame(sheet_list, columns=column_names)
        df = pd.DataFrame(sheet_list)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='summary_metrics', index=False)

    @staticmethod
    def save_mAP_plot(path_and_filename, map_05, map, map_075, title):
        """
        Saves the mAP@0.5 and mAP@0.5:0.95 per epoch.
        :param path_and_filename: path and filename to save the graphs.
        :param map_05: List containing mAP values at 0.5 IoU.
        :param map: List containing mAP values at 0.5:0.95 IoU.
        """
        figure = plt.figure(figsize=(10, 7), num=1, clear=True)
        ax = figure.add_subplot()
        ax.plot(
            map_05, color='tab:orange', linestyle='-', 
            label='mAP@0.5'
        )
        ax.plot(
            map_075, color='tab:blue', linestyle='-', 
            label='mAP@0.75'
        )
        ax.plot(
            map, color='tab:red', linestyle='-', 
            label='mAP@0.5:0.95'
        )
        ax.set_xlabel('Epochs')
        ax.set_ylabel('mAP')
        ax.legend()
        plt.title(title + LINE_FEED)  
        figure.savefig(f"{path_and_filename}")