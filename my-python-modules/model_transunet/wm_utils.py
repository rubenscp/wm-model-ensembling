"""
Project: White Mold 
Description: Utils methods and functions that manipulate images 
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 27/11/2023
Version: 1.0
"""
 
# Importing Python libraries 
import cv2
import os
import shutil
import pandas as pd

class WM_Utils:

    # Read image 
    @staticmethod
    def read_image(filename, path):
        path_and_filename = os.path.join(path, filename)
        image = cv2.imread(path_and_filename)
        return image

    # Save image
    @staticmethod
    def save_image(path_and_filename, image):
        cv2.imwrite(path_and_filename, image)

    # Create a folder
    @staticmethod
    def create_directory(folder):
        if not os.path.isdir(folder):    
            os.makedirs(folder)

    # Remove all files from a folder
    @staticmethod
    def remove_directory(folder):
        shutil.rmtree(folder, ignore_errors=True)

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

    @staticmethod
    def save_to_excel(path_and_filename, sheet_name, sheet_list):

        # creating dataframe from list        
        df = pd.DataFrame(sheet_list)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name=sheet_name, index=False)

    @staticmethod
    def save_losses(losses, path_and_filename):
        ''' 
        Save losses values into MSExcel file
        '''

        # preparing columns name to list
        column_names = [
            'epoch',
            'iteration',
            'loss',
            'loss_cros_entropy',
            'loss_dice'
        ]

        # creating dataframe from list 
        df = pd.DataFrame(losses, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='losses', index=False)

    # Draw bounding box in the image
    @staticmethod
    def draw_bounding_box(image, linP1, colP1, linP2, colP2,
                          background_box_color, thickness, label):
        
        # Start coordinate represents the top left corner of rectangle
        start_point = (colP1, linP1)

        # Ending coordinate represents the bottom right corner of rectangle
        end_point = (colP2, linP2)

        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(image, start_point, end_point, background_box_color, thickness)

        # setting the bounding box label
        font_scale = 0.5
        cv2.putText(image, label,
                    (colP1, linP1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, background_box_color, 2)

        # returning the image with bounding box drawn
        return image
