import numpy as np
import torch
from coco_eval import CocoEvaluator
# from tqdm.notebook import tqdm
from common.manage_log import *

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

def prepare_for_coco_detection(predictions):
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results        


def validate_the_model(parameters, device, model, processor, dataset_valid, dataloader_valid):

    # initialize evaluator with ground truth (gt)
    evaluator = CocoEvaluator(coco_gt=dataset_valid.coco, iou_types=["bbox"])
    
    logging_info(f'Running evaluation of the trained model ...')
    # for idx, batch in enumerate(tqdm(dataloader_valid)):
    for idx, batch in enumerate(dataloader_valid):

        # logging_info(f'')
        # logging_info(f'idx: {idx}')
        # logging_info(f'batch: {batch}')
        # logging_info(f'batch["pixel_values"].shape: {batch["pixel_values"].shape}')
        # logging_info(f'batch["pixel_mask"].shape: {batch["pixel_mask"].shape}')

        # get the inputs
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]] # these are in DETR format, resized + normalized
        # logging_info(f'labels: {labels}')

        # forward pass
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        # logging_info(f'outputs: {outputs}')
        # logging_info(f'outputs.keys(): {outputs.keys()}')
        # logging_info(f'outputs.logits.shape: {outputs.logits.shape}')
        # logging_info(f'outputs.pred_boxes.shape: {outputs.pred_boxes.shape}')
        # logging_info(f'outputs["logits"]: {outputs["logits"]}')
        # logging_info(f'outputs["pred_boxes"]: {outputs["pred_boxes"]}')

        # turn into a list of dictionaries (one item for each example in the batch)
        orig_target_sizes = torch.stack([target["orig_size"] for target in labels], dim=0)
        results = processor.post_process_object_detection(outputs, target_sizes=orig_target_sizes, threshold=0)
        # logging_info(f'labels: {labels}')
        # logging_info(f'results: {results}')

        # provide to metric
        # metric expects a list of dictionaries, each item
        # containing image_id, category_id, bbox and score keys
        predictions = {target['image_id'].item(): output for target, output in zip(labels, results)}
        predictions = prepare_for_coco_detection(predictions)
        evaluator.update(predictions)

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    # logging_info(f'evaluator: {evaluator}')

    return evaluator