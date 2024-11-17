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

# Importing python modules
from common.manage_log import *

# Importing entity classes
from common.entity.ImageAnnotation import ImageAnnotation

class ImageUtils:

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

    # Calculate coordinates of the new image to be cropped
    @staticmethod
    def calculate_coordinates_new_cropped_image(image_name_with_extension, 
                                                image, bounding_box, 
                                                crop_height_size, crop_width_size):
        # getting image shape        
        image_height = image.shape[0]
        image_width  = image.shape[1]

        # getting the height and width of the bounding box
        # bounding_box_height = bounding_box.lin_point2 - bounding_box.lin_point1
        # bounding_box_width  = bounding_box.col_point2 - bounding_box.col_point1

        # # evaluating if bounding box size is less than crop size 
        # # setting result with succes 
        # if bounding_box_height > crop_height_size  and  bounding_box_width > crop_width_size:
        #     text = f'Image {image_name_with_extension} with bounding box {bounding_box.id} of ({bounding_box_height},{bounding_box_width}) '
        #     text += f'is greater than cropping size ({crop_height_size},{crop_width_size})'
        #     logging_error(text)
        #     return False, 0, 0, 0, 0

        # calculating the coordinates of the new cropped image with the object positioned in 
        # the center of new image
        difference_height = crop_height_size - bounding_box.get_height()
        difference_width  = crop_width_size - bounding_box.get_width()
        half_of_difference_height = int(difference_height / 2.0)
        half_of_difference_width  = int(difference_width  / 2.0)

        # adjust_difference_heigth  = difference_height - (half_of_difference_height * 2) 
        # adjust_difference_width   = difference_width - (half_of_difference_width * 2)
        # half_of_difference_height += adjust_difference_heigth
        # half_of_difference_width  += adjust_difference_width

        # calculating the new coordinates of the cropped image 
        crop_lin_point1 = bounding_box.lin_point1 - half_of_difference_height
        crop_col_point1 = bounding_box.col_point1 - half_of_difference_width
        crop_lin_point2 = bounding_box.lin_point2 + half_of_difference_height
        crop_col_point2 = bounding_box.col_point2 + half_of_difference_width

        # checking size of new coordinates of cropped window image 
        crop_height_aux   = crop_lin_point2 - crop_lin_point1
        crop_width_aux    = crop_col_point2 - crop_col_point1
        difference_height = crop_height_size - crop_height_aux
        difference_width  = crop_width_size - crop_width_aux
        if difference_height == 1: crop_lin_point2 += 1
        if difference_width  == 1: crop_col_point2 += 1

        if difference_height != 1 and difference_height != 0:
            logging_error(f'Image_utils: {image_name_with_extension}-{bounding_box.id} difference height {difference_height}')
        if difference_width != 1 and difference_width != 0:
            logging_error(f'Image_utils: {image_name_with_extension}-{bounding_box.id} difference width {difference_width}')

        # evaluating the 8 (eigth) positions where the coordinates can be out of original image
            
        # position 1
        if crop_lin_point1 < 0                  and crop_col_point1 < 0   and \
        0 <= crop_lin_point2 <= image_height and 0 <= crop_col_point2 <= image_width:
            crop_lin_point1 = 0 
            crop_col_point1 = 0 
            crop_lin_point2 = crop_height_size
            crop_col_point2 = crop_width_size
            
        # position 2
        if crop_lin_point1 < 0                  and 0 <= crop_col_point1 <= image_width   and \
        0 <= crop_lin_point2 <= image_height and 0 <= crop_col_point2 <= image_width:
            crop_lin_point1 = 0 
            crop_lin_point2 = crop_height_size

        # position 3
        if crop_lin_point1 < 0                  and 0 <= crop_col_point1 <= image_width   and \
        0 <= crop_lin_point2 <= image_height and crop_col_point2 > image_width:
            crop_lin_point1 = 0 
            crop_col_point1 = image_width - crop_width_size 
            crop_lin_point2 = crop_height_size
            crop_col_point2 = image_width

        # position 4
        if 0 <= crop_lin_point1 <= image_height and crop_col_point1 < 0   and \
        0 <= crop_lin_point2 <= image_height and 0 <= crop_col_point2 <= image_width:
            crop_col_point1 = 0
            crop_col_point2 = crop_height_size

        # position 5
        if 0 <= crop_lin_point1 <= image_height and 0 <= crop_col_point1 <= image_width   and \
        0 <= crop_lin_point2 <= image_height and crop_col_point2 > image_width:
            crop_col_point1 = image_width - crop_height_size 
            crop_col_point2 = image_width

        # position 6
        if 0 <= crop_lin_point1 <= image_height and crop_col_point1 < 0   and \
        crop_lin_point2 > image_height       and 0 <= crop_col_point2 <= image_width:
            crop_lin_point1 = image_height - crop_height_size
            crop_col_point1 = 0 
            crop_lin_point2 = image_height
            crop_col_point2 = crop_width_size

        # position 7
        if 0 <= crop_lin_point1 <= image_height and 0 <= crop_col_point1 <= image_width   and \
        crop_lin_point2 > image_height       and 0 <= crop_col_point2 <= image_width:
            crop_lin_point1 = image_height - crop_height_size
            crop_lin_point2 = image_height
            
        # position 8
        if 0 <= crop_lin_point1 <= image_height and 0 <= crop_col_point1 <= image_width   and \
        crop_lin_point2 > image_height       and crop_col_point2 > image_width:
            crop_lin_point1 = image_height - crop_height_size
            crop_col_point1 = image_width - crop_width_size
            crop_lin_point2 = image_height
            crop_col_point2 = image_width

        # THIS CODE ISN'T NECESSARY 

        # # fine adjusting in the coordinates
        # height_difference = (crop_lin_point2 - crop_lin_point1) - crop_height_size
        # if height_difference != 0:
        #     crop_lin_point2 += height_difference * -1
        # width_difference = (crop_col_point2 - crop_col_point1) - crop_width_size
        # if width_difference != 0:
        #     crop_col_point2 += width_difference * -1

        # height_difference = (crop_lin_point2 - crop_lin_point1) % crop_height_size
        # if (crop_lin_point2 - crop_lin_point1) % crop_height_size != 0:
        #     crop_lin_point2 += 1
        # width_difference = (crop_col_point2 - crop_col_point1) % crop_width_size
        # if (crop_col_point2 - crop_col_point1) % crop_width_size != 0:
        #     crop_col_point2 += 1

        # setting result with succes 
        result = True 

        # returning the result and the coordinates of new image 
        return result, crop_lin_point1, crop_col_point1, crop_lin_point2, crop_col_point2
    
    # Draw bounding box in the image
    @staticmethod
    def draw_bounding_box(image, linP1, colP1, linP2, colP2,
                          background_box_color, thickness, label):
        # converting values to int
        linP1 = int(linP1)
        colP1 = int(colP1)
        linP2 = int(linP2)
        colP2 = int(colP2)
        
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

