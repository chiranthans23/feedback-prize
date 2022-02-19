
from config import CFG
from metric import *
from data import DataModule
from model import ModelModule

import numpy as np
import pandas as pd
import gc
import torch
from ast import literal_eval

from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# Weights and Biases
import wandb
wandb.login()

# W&B Logger
wandb_logger = WandbLogger(
    project='feedback-essay-lf',
    job_type='train',
    anonymous='allow',
    config=CFG().__dict__
)


if __name__ == '__main__':
    
    CFG = CFG()
    print(CFG.__dict__)
    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}

    train_text_df = pd.read_csv("../input/train_NER.csv")
    train_text_df.entities = train_text_df.entities.apply(lambda x: literal_eval(x) )
    train_df = pd.read_csv("../input/train.csv")

    # CHOOSE VALIDATION INDEXES
    IDS = train_df.id.unique()
    print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')

    # TRAIN VALID SPLIT 90% 10%
    train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
    
    # CREATE TRAIN SUBSET AND VALID SUBSET
    data = train_text_df[['id','text', 'entities']]
    train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['id','text', 'entities']].reset_index(drop=True)
    test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    
    train_gt = train_df.loc[train_df['id'].isin(IDS[train_idx])][['id','discourse_type', 'predictionstring']].reset_index(drop=True)
    test_gt = train_df.loc[train_df['id'].isin(IDS[valid_idx])][['id','discourse_type', 'predictionstring']].reset_index(drop=True)

    ###
    tag = 'ep{}-len{}'.format(CFG.epochs,CFG.max_length)

    CFG.steps_lr=len(train_dataset)//(CFG.batch_size) + 1

    tokenizer = AutoTokenizer.from_pretrained(CFG.tokpath,add_prefix_space=True)
    # help(ModelModule)
    dm = DataModule(train_dataset, test_dataset, tokenizer, CFG)
    model = ModelModule(CFG.__dict__,train_gt,test_gt)


    filename = f"{CFG.model_name}-{tag}"
    checkpoint_callback = ModelCheckpoint(monitor='val_f1', dirpath='./', mode='max', filename=filename,save_top_k=1)


    trainer = Trainer(
        gpus=CFG.n_procs,
        max_epochs=CFG.epochs,
        precision=CFG.precision,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        strategy="deepspeed_stage_3_offload"
    #    strategy=CFG.stg,
    #    log_every_n_steps=5,
        )

    trainer.fit(model, datamodule=dm)
        
    del model
    gc.collect()
    torch.cuda.empty_cache()