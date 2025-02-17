# matplotlib for visualization
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Importing python modules
from common.manage_log import *

# for visualization of images

# Function to visualize bounding boxes in the image
def plot_img_bbox(img, target, threshold=0.5, path_and_filename_image='', classes=None):

    # logging_info(f'plot_img_bbox - target: {target}')
    # Colors 
    # colors = [[0, 0, 0],        [255, 0, 0],        [0, 255, 0],    [0, 0, 255], 
    #           [238, 130, 238],  [106, 90, 205],     [188, 0, 239]]
    colors = ['black', 'red', 'green', 'yellow', 'blue', 'orange']
    
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    ind_score = 0
    for box in (target['boxes']):

        # evaluate "target" or "nms_prediction" dictionary
        if "scores" in target:
            scores = target["scores"].numpy()
            score = scores[ind_score]
            if score < threshold:
                # logging_info(f'score {score} less than threshold {threshold}')
                continue

        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                  width, height,
                                  linewidth = 2,
                                  edgecolor = 'r',
                                  facecolor = 'none',
                                  label='label')

        # Draw the bounding box on top of the image
        a.add_patch(rect)

        # evaluate "target" or "nms_prediction" dictionary
        if "scores" in target:
            scores = target["scores"].numpy()
            score = scores[ind_score]
            # logging_info(f'scores: {scores}')
            labels = target['labels'].numpy()
            class_id = labels[ind_score]
            # logging_info(f'classes: {classes}')
            # logging_info(f'class_id: {class_id}   classes[class_id]: {classes[class_id]}')
            class_name = classes[class_id]
            class_color = colors[class_id]
            # label_text = 'aphot ' + '%.2f' % score
            label_text = class_name + ' ('+ '%.2f' % score + ')'
            # a.annotate(label_text, (x,y-4), color='r', fontsize=11)
            a.annotate(label_text, (x,y-4), color=class_color, fontsize=11)

        ind_score += 1

    # saving image
    plt.savefig(path_and_filename_image)
    plt.show()
    plt.close()

# plotting the image with bboxes. Feel free to change the index
# img, target = dataset_to_check[1]
# plot_img_bbox(img, target, threshold=WHITE_MOLD_THRESHOLD, path_and_filename_image=local_results_faster_rcnn_predict_path + '/xxx.jpeg')
