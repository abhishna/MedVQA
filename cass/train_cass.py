import os
import numpy as np
import pytorch_lightning as pl
import torch
import pandas as pd
import timm
import math
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torchvision import transforms as tsfm
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchcontrib.optim import SWA
from torchmetrics import Metric
from torch.utils.tensorboard import SummaryWriter
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

TRAIN_IMGS_DIR = '../dataset/all_images/'
ALL_IMGS_LIST = os.listdir(TRAIN_IMGS_DIR)
BATCH_SIZE = 64
NUM_WORKERS = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

"""
Define train & valid image transformation
"""
DATASET_IMAGE_MEAN = (0.485, 0.456, 0.406)
DATASET_IMAGE_STD = (0.229, 0.224, 0.225)

train_transform = tsfm.Compose([tsfm.Resize((384,384)),
                                tsfm.RandomApply([tsfm.ColorJitter(0.2, 0.2, 0.2),tsfm.RandomPerspective(distortion_scale=0.2),], p=0.3),
                                tsfm.RandomApply([tsfm.ColorJitter(0.2, 0.2, 0.2),tsfm.RandomAffine(degrees=10),], p=0.3),
                                tsfm.RandomVerticalFlip(p=0.3),
                                tsfm.RandomHorizontalFlip(p=0.3),
                                tsfm.ToTensor(),
                                tsfm.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), ])

valid_transform = tsfm.Compose([tsfm.Resize((384,384)),
                                tsfm.ToTensor(),
                                tsfm.Normalize(DATASET_IMAGE_MEAN, DATASET_IMAGE_STD), ])

"""
Define dataset class
"""
class Dataset(Dataset):
    def __init__(self, train_imgs_dir: str, img_names: list, transform=None):
        self.img_dir = train_imgs_dir
        self.img_names = img_names
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        if self.img_names[idx].split('.')[-1] == 'dcm':
            img = Image.fromarray(read_xray(img_path)).convert('RGB')
        else:
            img = Image.open(img_path).convert('RGB')
        img_ts = self.transform(img)
        return img_ts

model_cnn = nn.DataParallel(timm.create_model('resnet50', pretrained=True))
model_vit = nn.DataParallel(timm.create_model('vit_base_patch16_384', pretrained=True))
model_cnn.to(device)
model_vit.to(device)

def ssl_train_model(train_loader,model_vit,optimizer_vit,scheduler_vit,model_cnn,optimizer_cnn,scheduler_cnn,num_epochs):
    writer = SummaryWriter()
    phase = 'train'
    model_cnn.train()
    model_vit.train()
    for i in tqdm(range(num_epochs)):
        with torch.set_grad_enabled(phase == 'train'):
            for img in train_loader:
                img = img.to(device)
                pred_vit = model_vit(img)
                pred_cnn = model_cnn(img)
                model_sim_loss=loss_fn(pred_vit,pred_cnn)
                loss = model_sim_loss.mean()
                loss.backward()

                nn.utils.clip_grad_norm_(model_vit.parameters(), 1.0)
                nn.utils.clip_grad_norm_(model_cnn.parameters(), 1.0)
                optimizer_cnn.step()
                optimizer_vit.step()
                scheduler_cnn.step()
                scheduler_vit.step()
            print('For -',i,'Loss:',loss)
            writer.add_scalar("Self-Supervised Loss/train", loss, i)
    writer.flush()

optimizer_cnn = SWA(torch.optim.Adam(model_cnn.parameters(), lr= 1e-4))
optimizer_vit = SWA(torch.optim.Adam(model_vit.parameters(), lr= 1e-4))
scheduler_cnn = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_cnn,
                                                                    T_max=16,
                                                                    eta_min=1e-6)
scheduler_vit = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vit,
                                                                    T_max=16,
                                                                    eta_min=1e-6)

def loss_fn(x, y):
    x =  torch.nn.functional.normalize(x, dim=-1, p=2)
    y =  torch.nn.functional.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

train_dataset = Dataset(TRAIN_IMGS_DIR, ALL_IMGS_LIST, train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

print('Training')
ssl_train_model(train_loader,model_vit,optimizer_vit,scheduler_vit,model_cnn,optimizer_cnn,scheduler_cnn,num_epochs=100)
#Saving SSL Models
print('Saving')
torch.save(model_cnn,'../models/cass-r50-isic.pt')
torch.save(model_vit,'../models/cass-r50-vit-isic.pt')