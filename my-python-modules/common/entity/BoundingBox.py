# BoundingBox Class

class BoundingBox:
    def __init__(self, id=None, class_id=None, class_title='', geometry_type='',
                 labeler_login='', created_at=None, updated_at=None, 
                 lin_point1=None, col_point1=None, lin_point2=None, col_point2=None,
                 confidence=0, annotation_filename=''):
        self.id = id
        self.class_id = class_id
        self.class_title = class_title
        self.geometry_type = geometry_type
        self.labeler_login = labeler_login
        self.created_at = created_at
        self.updated_at = updated_at
        self.lin_point1 = lin_point1
        self.col_point1 = col_point1
        self.lin_point2 = lin_point2
        self.col_point2 = col_point2
        self.confidence = confidence
        self.annotation_filename = annotation_filename

    def to_string(self):
        text = 'Class: ' + self.class_title + ' P1: (' + str(self.lin_point1) + ',' + str(
            self.col_point1) + ')  P2: (' + str(self.lin_point2) + ',' + str(
            self.col_point2) + ')' + '  confidence: ' + str(self.confidence)
        return text

    def get_height(self):
        return self.lin_point2 - self.lin_point1

    def get_width(self):
        return self.col_point2 - self.col_point1
    
    def get_area(self):
        return (self.get_height() * self.get_width())

    def get_id_class_SSD(self):
        if self.class_title == '__background__':
            return 0
        elif self.class_title == 'Apothecium':
            return 1
        elif self.class_title == 'Imature Sclerotium':
            return 2
        elif self.class_title == 'Mature Sclerotium':
            return 3
        elif self.class_title == 'White Mold':
            return 4
        elif self.class_title == 'Imature Sclerotium and White Mold':
            return 5
        elif self.class_title == 'Mushroom':
            return 6
        elif self.class_title == 'Petal':
            return 7
        elif self.class_title == 'Coal':
            return 8
        else:
            return -1       

    def evaluate_bbox_size_at_cropping_window(self, height_crop_window, width_crop_window):
        if self.get_height() <= height_crop_window and self.get_width() <= width_crop_window:
            return True, 'OK'
        if self.get_height() > height_crop_window  and self.get_width() > width_crop_window:
            return False, 'HW >'
        if self.get_height() > height_crop_window:
            return False, 'H >'
        if self.get_width() > width_crop_window:
            return False, 'W >'
            
    def check_consistency_of_coordinates(self, image_name, height, width):
        # initializing variable 
        text = ''

        # checking values of bounding box coordinates 
        if self.lin_point1 < 0:
            text +=  ' lin_point1: ' + str(self.lin_point1)
        if self.lin_point1 > height:
            text +=  ' lin_point1: ' + str(self.lin_point1)

        if self.col_point1 < 0:
            text +=  ' col_point1: ' + str(self.col_point1)
        if self.col_point1 > width:
            text +=  ' col_point1: ' + str(self.col_point1)

        if self.lin_point2 < 0:
            text +=  ' lin_point2: ' + str(self.lin_point2)
        if self.lin_point2 > height:
            text +=  ' lin_point2: ' + str(self.lin_point2)

        if self.col_point2 < 0:
            text +=  ' col_point2: ' + str(self.col_point2)
        if self.col_point2 > width:
            text +=  ' col_point2: ' + str(self.col_point2)

        # returning with correct consistency
        if text != '':
            # adding image name and bounding box id
            text = image_name + '-bbox-' + str(self.id) + ' ' + text 

        # returning result
        return text
        
    def get_box(self):
        box = [0, 0, 0, 0]
        # box[0] = self.lin_point1
        # box[1] = self.col_point1
        # box[2] = self.lin_point2
        # box[3] = self.col_point2

        box[0] = self.col_point1
        box[1] = self.lin_point1
        box[2] = self.col_point2
        box[3] = self.lin_point2
        
        # returning bounding box
        return box

    def get_class_index(self, classes, class_name):
        for index, value in enumerate(classes):
            if class_name == value:
                # success
                return index

        # return not found
        return None
