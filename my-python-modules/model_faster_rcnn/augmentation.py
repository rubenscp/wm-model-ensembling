# for image augmentations
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# Send train=True fro training transforms and False for val/test transforms
def get_transform(train):
    print(f'get_transform augmentation.py ')
    print(f'train: {train}')
    
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5), # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
            ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})