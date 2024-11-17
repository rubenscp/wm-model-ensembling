import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

from common.manage_log import *

class Detr(pl.LightningModule):     
    def __init__(self, lr, lr_backbone, weight_decay, 
                pretrained_model_name_or_path, cache_dir, num_labels,
                train_dataloader, val_dataloader):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        #  self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
        #                                                      revision="no_timm",
        #                                                      num_labels=len(id2label),
        #                                                      ignore_mismatched_sizes=True)
        # pretrained_model_name_or_path = '/home/lovelace/proj/proj939/rubenscp/research/white-mold-applications/wm-pretrained-model/detr-resnet-50'
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision="no_timm",
            num_labels=num_labels,
            ignore_mismatched_sizes=True, 
            local_files_only=True,
        )
        # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.train_dataloader = train_dataloader 
        self.val_dataloader = val_dataloader

    def forward(self, pixel_values, pixel_mask):
        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

        return outputs

    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
                {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": self.lr_backbone,
                },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                    weight_decay=self.weight_decay)

        return optimizer

    def train_dataloader(self):
        return self.train_dataloader

    def val_dataloader(self):
        return self.val_dataloader



def count_parameters(model):
    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)    
    return number_of_parameters

def count_layers(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        return 1
    return 0

def compute_num_layers(model):
    num_layers = sum(count_layers(layer) for layer in model.modules())
    return num_layers
      