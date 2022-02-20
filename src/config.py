from tqdm.auto import tqdm
import os
import random
import numpy as np
import pandas as pd

import gc
pd.set_option('display.max_columns', None)
gc.enable()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
#from torch.utils.data import RandomSampler, SequentialSampler,TensorDataset
from torch.optim.lr_scheduler import OneCycleLR#,CosineAnnealingLR
#from torch.optim import lr_scheduler

from pytorch_lightning import LightningModule, LightningDataModule,Trainer
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor

# transformer
from transformers import AutoTokenizer, AutoModel, AdamW,AutoConfig,AutoModelForTokenClassification

#
from sklearn.metrics import accuracy_score



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    
    
class CFG:
    def __init__(self):
        self.n_procs=1
        self.num_workers=4
        self.precision = 16
        self.seed=2022
        self.scheduler='lr_logging'
        self.model_name='BigBird'
        self.modelpath='../input/py-bigbird-v26'
        self.tokpath = '../input/py-bigbird-v26'
        self.max_length=1024
        self.num_labels=15
        self.epochs=2
        self.batch_size = 10
        self.val_batch_size = self.batch_size 
        self.lr=2.5e-5
        self.max_lr=self.lr
        self.steps_lr=123//(self.batch_size) + 1  
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(self.seed)
