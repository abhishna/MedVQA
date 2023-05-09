from dataset import VQADataset
from models.baseline import VQABaseline
from PIL import Image
from torch.optim import Adadelta, Adam, lr_scheduler, RMSprop
from torch.utils.data import Dataset, DataLoader
from train import *
from utils import parse_tb_logs, get_model

train_ds     = VQADataset("/home/an3729/datasets", top_k = 10, max_length = 50, transform = None,
                                  use_image_embedding = True, 
                                  ignore_unknowns = True)

val_ds       = VQADataset("/home/an3729/datasets", top_k = 10, max_length = 50, transform = None,
                                  use_image_embedding = True, 
                                  ignore_unknowns = True, mode = 'val')
batch_size = 384
train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle = True, num_workers = 2, pin_memory = True)
val_loader   = DataLoader(val_ds, batch_size = batch_size, num_workers = 2, pin_memory = True)
c=0
for step, (images, questions, answers) in enumerate(train_loader):
    c+=1
    print(answers)
    continue
print(c)
print("done")
