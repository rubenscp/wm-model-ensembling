import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_transunet.datasets_tun.dataset_synapse import Synapse_dataset
from model_transunet.utils import test_single_volume, test_single_image_white_mold
from model_transunet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from model_transunet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from torchinfo import summary
from ptflops import get_model_complexity_info

from model_transunet.datasets_tun.dataset_white_mold import WhiteMold_dataset

from common.metrics import *


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')

parser.add_argument('--max_iterations', type=int,default=20000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=30, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')

parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str, default='ViT-B_16', help='select one vit model')

parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--vit_patches_size', type=int, default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()


def inference(args, model, test_save_path=None):
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        print(f'inference image shape: {image.shape}')
        print(f'inference label shape: {label.shape}')
        print(f'inference case_name: {case_name}')
                
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

# the function takes the original prediction and the iou threshold.
# this technique requires the predicted bounding boxes and their predicted scores, 
# beyond the IoU thershold
# def apply_nms(orig_prediction, iou_thresh=0.3):

#     # logging_info(f'before apply nms')
#     # logging_info(f'orig_prediction: {orig_prediction}')

#     # torchvision returns the indices of the bboxes to keep
#     keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

#     # logging_info(f'after apply nms')
#     # logging_info(f'keep: {keep}')

#     final_prediction = orig_prediction
#     final_prediction['boxes'] = final_prediction['boxes'][keep]
#     final_prediction['scores'] = final_prediction['scores'][keep]
#     final_prediction['labels'] = final_prediction['labels'][keep]

#     return final_prediction


def inference_white_mold(args, model, parameters, test_save_path=None):
    # print(f'inferecen_white_mold 1 - args: {args}')    
    # print(f'inferecen_white_mold 2 - test_save_path: {test_save_path}')

    # creating predictions dictionary of all images 
    all_predictions = {}

    # setting tes image folder 
    test_image_folder = os.path.join(args.volume_path, "test")
    print(f'test_image_folder: {test_image_folder}')

    # loading test image dataset 
    db_test = args.Dataset(base_dir=args.volume_path, split="test", list_dir=args.list_dir)

    # loading test image loader 
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    logging.info("{} test iterations per epoch".format(len(testloader)))
    print("{} test iterations per epoch".format(len(testloader)))

    # creating metric object 
    inference_metric = Metrics(
        model='transunet',
        # number of classes doesn't consider the background class
        number_of_classes=5,
    )

    # creating tested folder to save inferenceed images 
    tested_folder = os.path.join('/home/lovelace/proj/proj939/rubenscp/', test_save_path, 'tested-image')
    print(f'tested_folder: {tested_folder}')
    Utils.create_directory(tested_folder)

    model.eval()
    metric_list = 0.0
    # count = 0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):  

        # if count > 10: 
        #     break  

        print(f'sampled_batch["image"].shape: {sampled_batch["image"].shape}')
        print(f'sampled_batch["image"].size()[2:]): {sampled_batch["image"].size()[2:]}')
        print(f'len(sampled_batch["image"].shape.size()[2:]): {len(sampled_batch["image"].size()[2:])}')
        # sampled_batch["image"].shape: torch.Size([1, 1, 300, 300, 3])
        if len(sampled_batch["image"].size()[2:]) == 2:
            h, w = sampled_batch["image"].size()[2:]
            print(f'h:{h} w:{w}')
        else:
            h, w, channels = sampled_batch["image"].size()[2:]
            print(f'h:{h} w:{w} channels:{channels}')

        image, label, case_name, original_image, annotated_bounding_boxes = \
                sampled_batch['image'], \
                sampled_batch['label'], \
                sampled_batch['case_name'][0], \
                sampled_batch['original_image'], \
                sampled_batch['bbox']

        metric_i = test_single_image_white_mold(image, label, model, classes=args.num_classes, 
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, 
                                      z_spacing=args.z_spacing, dataset=args.dataset, 
                                      original_image=original_image, 
                                      annotated_bounding_boxes=annotated_bounding_boxes,
                                      inference_metric=inference_metric, 
                                      tested_folder=tested_folder, 
                                      test_image_folder=test_image_folder,
                                      all_predictions=all_predictions,
                                      )

        print(f'test.py - inference_metric: {inference_metric.to_string()}')

        metric_list += np.array(metric_i)
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

        # print(f'test.py - all_predictions: {len(all_predictions)} - {all_predictions}')
        
    # metric_list = metric_list / len(db_test)
    # for i in range(1, args.num_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
 
    # setting classes name 
    classes = ["__background__", 
               "Apothecium", 
               "Imature Sclerotium", 
               "Mature Sclerotium", 
               "White Mold", 
               "Imature Sclerotium and White Mold"
               ]

    # Computing Confusion Matrix 
    # model_name = 'TransUNet'
    model_name = parameters['model_transunet']['neural_network_model']['model_name']
    num_classes = 6
    threshold = parameters['model_transunet']['neural_network_model']['threshold']
    iou_threshold = parameters['model_transunet']['neural_network_model']['iou_threshold']
    non_maximum_suppression =  parameters['model_transunet']['neural_network_model']['non_maximum_suppression']
    metrics_folder = os.path.join('/home/lovelace/proj/proj939/rubenscp/', test_save_path, 'metrics')
    print(f'metrics_folder: {metrics_folder}')
    Utils.create_directory(metrics_folder)
    running_id_text = '001'    
    # tested_folder = os.path.join('/home/lovelace/proj/proj939/rubenscp/', test_save_path, 'tested-image')
    # print(f'tested_folder: {tested_folder}')
    # Utils.create_directory(tested_folder)
    inference_metric.compute_confusion_matrix(model_name, num_classes, threshold, iou_threshold, 
                                              metrics_folder, running_id_text, tested_folder)
    inference_metric.confusion_matrix_to_string()

    logging_info(f'inference_metric.to_string(): {inference_metric.to_string()}')
  
    # saving confusion matrix plots
    title =  'Full Confusion Matrix' + \
             ' - Model: ' + model_name + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(threshold) + \
             '   IoU threshold: ' + str(iou_threshold) + \
             '   Non-maximum Supression: ' + str(non_maximum_suppression)
    path_and_filename = os.path.join(
        metrics_folder, model_name + '_' + running_id_text + '_confusion_matrix_full.png')
    number_of_classes = num_classes
    cm_classes = classes[0:(number_of_classes+1)]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Incorrect predictions')    
    y_labels_names.append('Undetected objects')
    format='.0f'

    logging_info(f'inference_metric.full_confusion_matrix: {inference_metric.full_confusion_matrix}')
    logging_info(f'path_and_filename: {path_and_filename}')
    logging_info(f'title: {title}')
    logging_info(f'format: {format}')
    logging_info(f'x_labels_names: {x_labels_names}')
    logging_info(f'y_labels_names: {y_labels_names}')
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        metrics_folder,
        model_name + '_' + running_id_text + '_confusion_matrix_full.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix,
                                      path_and_filename, 
                                      x_labels_names, y_labels_names,
                                      inference_metric.tp_per_class,
                                      inference_metric.fp_per_class,
                                      inference_metric.fn_per_class,
                                      inference_metric.tn_per_class
    )

    title =  'Full Confusion Matrix Normalized' + \
             ' - Model: ' + model_name + \
             '   # images:' + str(inference_metric.confusion_matrix_summary['number_of_images'])
    title += LINE_FEED + \
             'Confidence threshold: ' + str(threshold) + \
             '   IoU threshold: ' + str(iou_threshold) + \
             '   Non-maximum Supression: ' + str(non_maximum_suppression)
    path_and_filename = os.path.join(
        metrics_folder, model_name + '_' + running_id_text + '_confusion_matrix_full_normalized.png'
    )
    cm_classes = classes[0:(number_of_classes+1)]
    x_labels_names = cm_classes.copy()
    y_labels_names = cm_classes.copy()
    x_labels_names.append('Incorrect predictions')    
    y_labels_names.append('Undetected objects')
    format='.2f'
    Utils.save_plot_confusion_matrix(inference_metric.full_confusion_matrix_normalized, 
                                     path_and_filename, title, format,
                                     x_labels_names, y_labels_names)
    path_and_filename = os.path.join(
        metrics_folder, model_name + '_' + running_id_text + '_confusion_matrix_full_normalized.xlsx'
    )
    Utils.save_confusion_matrix_excel(inference_metric.full_confusion_matrix_normalized,
                                      path_and_filename,
                                      x_labels_names, y_labels_names,
                                      inference_metric.tp_per_class,
                                      inference_metric.fp_per_class,
                                      inference_metric.fn_per_class,
                                      inference_metric.tn_per_class
    )
    
    # saving metrics from confusion matrix
    path_and_filename = os.path.join(
        metrics_folder, model_name + '_' + running_id_text + '_confusion_matrix_metrics.xlsx'
    )
    
    sheet_name='metrics_summary'
    sheet_list = []
    sheet_list.append(['Metrics Results calculated by application', ''])
    sheet_list.append(['', ''])
    sheet_list.append(['Model', f'{model_name}'])
    sheet_list.append(['', ''])
    sheet_list.append(['Threshold',  f"{threshold:.2f}"])
    sheet_list.append(['IoU Threshold',  f"{iou_threshold:.2f}"])
    sheet_list.append(['Non-Maximum Supression',  f"{non_maximum_suppression:.2f}"])
    sheet_list.append(['', ''])

    sheet_list.append(['TP / FP / FN / TN per Class', ''])
    cm_classes = classes[1:(number_of_classes+1)]

    # setting values of TP, FP, and FN per class
    sheet_list.append(['Class', 'TP', 'FP', 'FN', 'TN'])
    for i, class_name in enumerate(classes[1:(number_of_classes+1)]):
    # for i, class_name in enumerate(classes):
        row = [class_name, 
               f'{inference_metric.tp_per_class[i]:.0f}',
               f'{inference_metric.fp_per_class[i]:.0f}',
               f'{inference_metric.fn_per_class[i]:.0f}',
               f'{inference_metric.tn_per_class[i]:.0f}',
              ]
        sheet_list.append(row)

    i += 1
    row = ['Total',
           f'{inference_metric.tp_model:.0f}',
           f'{inference_metric.fp_model:.0f}',
           f'{inference_metric.fn_model:.0f}',
           f'{inference_metric.tn_model:.0f}',
          ]
    sheet_list.append(row)    
    sheet_list.append(['', ''])

    # setting values of metrics precision, recall, f1-score and dice per class
    sheet_list.append(['Class', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Dice'])
    for i, class_name in enumerate(classes[1:(number_of_classes+1)]):
    # for i, class_name in enumerate(classes):
        row = [class_name, 
               f'{inference_metric.accuracy_per_class[i]:.8f}',
               f'{inference_metric.precision_per_class[i]:.8f}',
               f'{inference_metric.recall_per_class[i]:.8f}',
               f'{inference_metric.f1_score_per_class[i]:.8f}',
               f'{inference_metric.dice_per_class[i]:.8f}',
              ]
        sheet_list.append(row)

    i += 1
    row = ['Model Metrics',
               f'{inference_metric.get_model_accuracy():.8f}',
               f'{inference_metric.get_model_precision():.8f}',
               f'{inference_metric.get_model_recall():.8f}',
               f'{inference_metric.get_model_f1_score():.8f}',
               f'{inference_metric.get_model_dice():.8f}',
          ]
    sheet_list.append(row)
    sheet_list.append(['', ''])

    # metric measures 
    sheet_list.append(['Metric measures', ''])
    sheet_list.append(['number_of_images', f'{inference_metric.confusion_matrix_summary["number_of_images"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_target"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_predicted', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted"]:.0f}'])
    sheet_list.append(['number_of_bounding_boxes_predicted_with_target', f'{inference_metric.confusion_matrix_summary["number_of_bounding_boxes_predicted_with_target"]:.0f}'])
    sheet_list.append(['number_of_incorrect_predictions', f'{inference_metric.confusion_matrix_summary["number_of_ghost_predictions"]:.0f}'])
    sheet_list.append(['number_of_undetected_objects', f'{inference_metric.confusion_matrix_summary["number_of_undetected_objects"]:.0f}'])

    # saving metrics sheet
    Utils.save_metrics_excel(path_and_filename, sheet_name, sheet_list)
    logging_sheet(sheet_list)

    # returning all predictions 
    return all_predictions


def show_model_size(model):
    print(f'')
    print(f'Showing Model size')
    print(f'')

    # Get the model summary to see size in MB and parameter details
    summary(model, input_size=(1, 3, 224, 224))  # Adjust input_size based on your input
    
    # Calculate GFLOPS and parameters
    with torch.cuda.device(0):
        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
        print(f'')
        print(f"GFLOPS: {flops}, Parameters: {params}")
        print(f'')


# if __name__ == "__main__":
def inference_neural_network_model(parameters, device):
    
    # setting parameter values as it was called by a prompt command
    # to process the White Mold image dataset 
    args.dataset = 'WhiteMold'
    args.vit_name = 'R50-ViT-B_16'
    args.max_epochs = 150
    args.volume_path = 'research/white-mold-applications/project_TransUNet/data/Synapse/test_vol_h5'
    args.list_dir = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-ensembling/model_transunet/lists/lists_Synapse'

    args.is_savenii = True
    args.test_save_dir = 'research/white-mold-applications/model/predictions'

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': 'research/white-mold-applications/project_TransUNet/data/Synapse/test_vol_h5',
            'list_dir': '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-model-ensembling/my-python-modules/model_transunet/lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'WhiteMold': {
            'Dataset': WhiteMold_dataset,
            'volume_path': '/home/lovelace/proj/proj939/rubenscp/research/white-mold-dataset/results-pre-processed-images/running-0021-15ds-300x300-merged-classes/splitting_by_images/4-balanced-output-dataset/mask-image',
            'list_dir': '',
            'num_classes': 8,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # name the same snapshot defined in train script!
    args.exp = 'TU_' + dataset_name + str(args.img_size)
    snapshot_path = "research/white-mold-applications/model/{}/{}".format(args.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    if dataset_name == 'ACDC':  # using max_epoch instead of iteration to control training duration
        snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(args.img_size)
    snapshot_path = snapshot_path + '_s'+str(args.seed) if args.seed!=1234 else snapshot_path

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot = os.path.join(snapshot_path, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))
    net.load_state_dict(torch.load(snapshot))
    print(f'Model loaded from {snapshot}')   
    snapshot_name = snapshot_path.split('/')[-1]

    log_folder = './test_log/test_log_' + args.exp
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/'+snapshot_name+".txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    print(f'arg.is_savenii: {args.is_savenii} - before inference')
    if args.is_savenii:
        print(f'arg.is_savenii: {args.is_savenii} - if true')
        # args.test_save_dir = '../predictions'
        print(f'args.test_save_dir: {args.test_save_dir}')
        print(f'args.exp: {args.exp}')
        print(f'snapshot_name: {snapshot_name}')
        test_save_path = os.path.join(args.test_save_dir, args.exp, snapshot_name)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        print(f'arg.is_savenii: {args.is_savenii} - if false')
        test_save_path = None

    print(f'model: {net}')
    print(f'test_save_path: {test_save_path}')

    show_model_size(net)

    all_predictions = inference_white_mold(args, net, parameters, test_save_path)

    logging.info(f'Inference white mold dataset finished!')

    return all_predictions

