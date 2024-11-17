"""
Project: White Mold 
Description: Class to manage statistics of annotations image dataset
Author: Rubens de Castro Pereira
Advisors: 
    Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
    Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
    Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans
Date: 23/02/2024
Version: 1.1
"""

# Importing python modules
from common.utils import *
from common.manage_log import * 
from common.entity.ImageAnnotation import ImageAnnotation

import pandas as pd

# ###########################################
# Constants
# ###########################################
LINE_FEED = '\n'

class AnnotationsStatistic:

    def __init__(self):
        self.annotations_statistic = {}

    def to_string(self):
        pretty_json = Utils.get_pretty_json(self.annotations_statistic)
        return pretty_json

    def processing_statistics(self, parameters, steps):

        # getting parameters values 
        annotation_format = parameters['input']['input_dataset']['annotation_format']
        input_image_size = parameters['input']['input_dataset']['input_image_size']
        classes =  parameters['neural_network_model']['classes']
        image_dataset_folder_train = parameters['processing']['image_dataset_folder_train']
        image_dataset_folder_valid = parameters['processing']['image_dataset_folder_valid']
        image_dataset_folder_test  = parameters['processing']['image_dataset_folder_test']
        
        # initilizing statistics
        self.initialize_statistics(annotation_format, input_image_size, classes)

        # selecting annotation format to get statistics 
        if parameters['input']['input_dataset']['annotation_format'] == 'ssd':
            x = 0

        elif parameters['input']['input_dataset']['annotation_format'] == 'ssd_pascal_voc':
            self.process_ssd_pascal_voc_format(
                annotation_format, input_image_size, classes,
                image_dataset_folder_train,
                image_dataset_folder_valid, 
                image_dataset_folder_test,
                steps
            )
            # computing total percentage per step
            self.compute_total_percentage(annotation_format, input_image_size)

        elif parameters['input']['input_dataset']['annotation_format'] == 'faster_rcnn':
            self.process_ssd_pascal_voc_format(
                            annotation_format, input_image_size, classes,
                            image_dataset_folder_train,
                            image_dataset_folder_valid, 
                            image_dataset_folder_test,
                            steps
                        )
            # computing total percentage per step
            self.compute_total_percentage(annotation_format, input_image_size)

        elif parameters['input']['input_dataset']['annotation_format'] == 'yolov8':
            # setting specific folders of the image annotation labels 
            image_dataset_folder_train = os.path.join(image_dataset_folder_train, 'labels')
            image_dataset_folder_valid = os.path.join(image_dataset_folder_valid, 'labels')
            image_dataset_folder_test = os.path.join(image_dataset_folder_test, 'labels')

            self.process_yolov8_format(
                            annotation_format, input_image_size, classes,
                            image_dataset_folder_train,
                            image_dataset_folder_valid, 
                            image_dataset_folder_test,
                            steps
                        )
            # computing total percentage per step
            self.compute_total_percentage(annotation_format, input_image_size)


    def initialize_statistics(self, annotation_format, input_image_size, classes):

        # setting statistic dimensions
        self.annotations_statistic[annotation_format] = {}
        self.annotations_statistic[annotation_format][input_image_size] = {}
        self.annotations_statistic[annotation_format][input_image_size]['train'] = {}
        self.annotations_statistic[annotation_format][input_image_size]['valid'] = {}
        self.annotations_statistic[annotation_format][input_image_size]['test'] = {}                

        for class_name in classes:
            if class_name[:5] != 'class' and class_name[:14] != '__background__':                
                self.annotations_statistic[annotation_format][input_image_size]['train'][class_name] = {}
                self.annotations_statistic[annotation_format][input_image_size]['train'] \
                                          [class_name]['number_of'] = 0
                self.annotations_statistic[annotation_format][input_image_size]['train'] \
                                          [class_name]['percentage'] = 0.0
                self.annotations_statistic[annotation_format][input_image_size]['valid'][class_name] = {}
                self.annotations_statistic[annotation_format][input_image_size]['valid'] \
                                          [class_name]['number_of'] = 0
                self.annotations_statistic[annotation_format][input_image_size]['valid'] \
                                          [class_name]['percentage'] = 0.0
                self.annotations_statistic[annotation_format][input_image_size]['test'][class_name] = {}
                self.annotations_statistic[annotation_format][input_image_size]['test'] \
                                          [class_name]['number_of'] = 0   
                self.annotations_statistic[annotation_format][input_image_size]['test'] \
                                          [class_name]['percentage'] = 0.0

        # creating totals
        self.annotations_statistic[annotation_format][input_image_size]['train']['total'] = {}
        self.annotations_statistic[annotation_format][input_image_size] \
                                  ['train']['total']['number_of'] = 0
        self.annotations_statistic[annotation_format][input_image_size] \
                                  ['train']['total']['percentage'] = 0.0
        self.annotations_statistic[annotation_format][input_image_size]['valid']['total'] = {}
        self.annotations_statistic[annotation_format][input_image_size] \
                                  ['valid']['total']['number_of'] = 0
        self.annotations_statistic[annotation_format][input_image_size] \
                                  ['valid']['total']['percentage'] = 0.0
        self.annotations_statistic[annotation_format][input_image_size]['test']['total'] = {}
        self.annotations_statistic[annotation_format][input_image_size] \
                                  ['test']['total']['number_of'] = 0   
        self.annotations_statistic[annotation_format][input_image_size] \
                                  ['test']['total']['percentage'] = 0.0 

    def process_ssd_pascal_voc_format(self,
        annotation_format, input_image_size, classes, 
        image_dataset_folder_train, 
        image_dataset_folder_valid, 
        image_dataset_folder_test, 
        steps
    ):

        # get annotations list 
        if 'train' in steps:
            train_annotations_list = Utils.get_files_with_extensions(
                image_dataset_folder_train, '.xml'
            )
            input_dataset_type = 'train'
            self.process_image_annotations_xml(
                annotation_format, input_image_size, input_dataset_type, classes, 
                image_dataset_folder_train, train_annotations_list
            )

        if 'valid' in steps:
            valid_annotations_list = Utils.get_files_with_extensions(
                image_dataset_folder_valid, '.xml'
            )
            input_dataset_type = 'valid'
            self.process_image_annotations_xml(
                annotation_format, input_image_size, input_dataset_type, classes, 
                image_dataset_folder_valid, valid_annotations_list
            )

        if 'test' in steps:
            test_annotations_list = Utils.get_files_with_extensions(
                image_dataset_folder_test, '.xml'
            )
            input_dataset_type = 'test'
            self.process_image_annotations_xml(
                annotation_format, input_image_size, input_dataset_type, classes, 
                image_dataset_folder_test, test_annotations_list
            )

    def process_yolov8_format(self,
        annotation_format, input_image_size, classes, 
        image_dataset_folder_train, 
        image_dataset_folder_valid, 
        image_dataset_folder_test,
        steps
    ):

        # get annotations list 
        if 'train' in steps:
            train_annotations_list = Utils.get_files_with_extensions(
                image_dataset_folder_train, '.txt'
            )
            input_dataset_type = 'train'
            self.process_image_annotations_yolo(
                annotation_format, input_image_size, input_dataset_type, classes, 
                image_dataset_folder_train, train_annotations_list
            )

        if 'valid' in steps:
            valid_annotations_list = Utils.get_files_with_extensions(
                image_dataset_folder_valid, '.txt'
            )
            input_dataset_type = 'valid'
            self.process_image_annotations_yolo(
                annotation_format, input_image_size, input_dataset_type, classes, 
                image_dataset_folder_valid, valid_annotations_list
            )

        if 'test' in steps:
            test_annotations_list = Utils.get_files_with_extensions(
                image_dataset_folder_test, '.txt'
            )
            input_dataset_type = 'test'
            self.process_image_annotations_yolo(
                annotation_format, input_image_size, input_dataset_type, classes, 
                image_dataset_folder_test, test_annotations_list
            )


    def compute_total_percentage(self, annotation_format, input_image_size):
        total_number_of_images = \
            self.annotations_statistic[annotation_format][input_image_size]['train']['total']['number_of'] + \
            self.annotations_statistic[annotation_format][input_image_size]['valid']['total']['number_of'] + \
            self.annotations_statistic[annotation_format][input_image_size]['test']['total']['number_of'] 

        self.annotations_statistic[annotation_format][input_image_size]['train']['total']['percentage'] = \
            self.annotations_statistic[annotation_format][input_image_size]['train']['total']['number_of'] / total_number_of_images * 100
        self.annotations_statistic[annotation_format][input_image_size]['valid']['total']['percentage'] = \
            self.annotations_statistic[annotation_format][input_image_size]['valid']['total']['number_of'] / total_number_of_images * 100
        self.annotations_statistic[annotation_format][input_image_size]['test']['total']['percentage'] = \
            self.annotations_statistic[annotation_format][input_image_size]['test']['total']['number_of'] / total_number_of_images * 100


    # ###########################################
    # Methods of Level 2
    # ###########################################

    def process_image_annotations_xml(self,
        annotation_format, 
        input_image_size, 
        input_dataset_type,
        classes,
        image_dataset_folder,
        annotations_list,
    ):

        # logging process
        logging_info(f'Processing image dataset of {input_dataset_type} with {len(annotations_list)} images')

        # setting total 
        self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                  ['total']['number_of'] = len(annotations_list)
        self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                  ['total']['percentage'] = 100

        # processing annotations list 
        for annotation_file in annotations_list:
            path_and_filename_xml_annotation = os.path.join(image_dataset_folder, annotation_file)

            # getting all annotations of the image 
            image_annotation = ImageAnnotation()
            image_annotation.get_annotation_file_in_voc_pascal_format(path_and_filename_xml_annotation)

            # processing bounding boxes 
            for bounding_box in image_annotation.bounding_boxes:
                counter = self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                                    [bounding_box.class_title]['number_of']
                counter += 1 
                self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                          [bounding_box.class_title]['number_of'] = counter

                # updating percentage 
                new_percentage = counter / self.annotations_statistic[annotation_format][input_image_size] \
                                 [input_dataset_type]['total']['number_of'] * 100.0

                self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                          [bounding_box.class_title]['percentage'] = new_percentage


    def process_image_annotations_yolo(self,
            annotation_format, 
            input_image_size, 
            input_dataset_type,
            classes,
            image_dataset_folder,
            annotations_list,
        ):

            # logging process
            logging_info(f'Processing image dataset of {input_dataset_type} with {len(annotations_list)} images')

            # setting total 
            self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                    ['total']['number_of'] = len(annotations_list)
            self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                    ['total']['percentage'] = 100

            # processing annotations list 
            for annotation_file in annotations_list:
                path_and_filename_yolo_annotation = os.path.join(image_dataset_folder, annotation_file)

                print()
                print(f'rubens path_and_filename_yolo_annotation: {path_and_filename_yolo_annotation}')

                # getting all annotations of the image 
                image_annotation = ImageAnnotation()
                height = width = input_image_size
                image_annotation.get_annotation_file_in_yolo_v5_format(path_and_filename_yolo_annotation, classes, height, width)

                # processing bounding boxes 
                for bounding_box in image_annotation.bounding_boxes:

                    print(f'rubens error bounding_box.class_title: {bounding_box.class_title}')

                    counter = self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                                        [bounding_box.class_title]['number_of']
                    counter += 1 
                    self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                            [bounding_box.class_title]['number_of'] = counter

                    # updating percentage 
                    new_percentage = counter / self.annotations_statistic[annotation_format][input_image_size] \
                                    [input_dataset_type]['total']['number_of'] * 100.0

                    self.annotations_statistic[annotation_format][input_image_size][input_dataset_type] \
                                            [bounding_box.class_title]['percentage'] = new_percentage

    def save_annotations_statistics(self, 
                                    path_and_filename,
                                    annotation_format, 
                                    input_image_size, 
                                    classes,
        ):

        # preparing columns name to list
        column_names = [
            'classes',
            'training # images',            
            'training percentage',
            'validating # images',            
            'validating percentage',
            'testing # images',            
            'testing percentage',
        ]

        list = []
        classes.append('total')
        for class_item  in classes:
        # for annotation in self.annotations_statistic:
            item = []
            item.append(class_item)
            item.append(self.annotations_statistic \
                        [annotation_format][input_image_size]['train'][class_item]['number_of'])
            item.append(self.annotations_statistic \
                        [annotation_format][input_image_size]['train'][class_item]['percentage'])
            item.append(self.annotations_statistic \
                        [annotation_format][input_image_size]['valid'][class_item]['number_of'])
            item.append(self.annotations_statistic \
                        [annotation_format][input_image_size]['valid'][class_item]['percentage'])
            item.append(self.annotations_statistic \
                        [annotation_format][input_image_size]['test'][class_item]['number_of'])
            item.append(self.annotations_statistic \
                        [annotation_format][input_image_size]['test'][class_item]['percentage'])
            list.append(item)                

        # creating dataframe from list 
        df = pd.DataFrame(list, columns=column_names)

        # writing excel file from dataframe
        df.to_excel(path_and_filename, sheet_name='statistics annotations', index=False)
