
from config import CFG
import numpy as np
import pandas as pd

class Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_wids):
        #super().__init__()
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids # for validation
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        LABEL_ALL_SUBTOKENS=True
        
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT (use is_split_into_words)
        encoding = self.tokenizer(text.split(),
                             is_split_into_words=True,
                             #return_offsets_mapping=True, 
                             padding='max_length', 
                             truncation=True, 
                             max_length=self.max_len)
        
        # padding and prefix=None
        # map token[0,0,0,1,2] to split['a.b','c','d']
        word_ids = encoding.word_ids()  
        
        # CREATE TARGETS
        if not self.get_wids:
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids: # tokens len                        
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx: # change word   
                    label_ids.append( labels_to_ids[word_labels[word_idx]] )
                else: # same word
                    if LABEL_ALL_SUBTOKENS:
                        label_ids.append( labels_to_ids[word_labels[word_idx]] )
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx
        
        if self.get_wids:
            word_ids2 = [w if w is not None else -1 for w in word_ids]
            return {
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(label_ids, dtype=torch.long),
                'wids': torch.tensor(word_ids2, dtype=torch.long)
            }
        else:
            return {
                'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
                'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
                'labels': torch.tensor(label_ids, dtype=torch.long)
            }


class DataModule(LightningDataModule):
    def __init__(self, train_df, val_df, tokenizer, get_wids=None, cfg=None):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.get_wids = get_wids

    
    def setup(self,stage):
        if stage == 'fit':
            self.train_ds = Dataset(self.train_df, self.tokenizer, self.cfg.max_length, self.get_wids)
            self.valid_ds = Dataset(self.val_df,   self.tokenizer, self.cfg.max_length, self.get_wids)

    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, 
            shuffle=True, num_workers=self.cfg.num_workers,
            pin_memory=True
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.cfg.val_batch_size, 
            shuffle=False, num_workers=self.cfg.num_workers,
            pin_memory=True
            )


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        #self.save_hyperparameters(cfg)
        config = AutoConfig.from_pretrained(self.cfg['modelpath'])
        self.model = AutoModelForTokenClassification.from_pretrained(self.cfg['modelpath'],config=config)
        #self.model = AutoModel.from_pretrained(self.hparams.modelpath,config=config)
       
        #self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask,labels):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        return out.loss,out.logits
    
    def training_step(self, batch, batch_idx):
        loss,logits = self(batch['input_ids'], batch['attention_mask'],batch['labels'])
        
        labels = batch['labels']
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.cfg['num_labels']) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        #tr_labels.extend(labels)
        #tr_preds.extend(predictions)
        
        try:
            score = accuracy_score(labels.cpu().numpy(), predictions.cpu().detach().numpy())
        except ValueError:
            score=0.0
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_score', score, on_step=False, on_epoch=True, prog_bar=True)
        return {'loss': loss, 'train_score': score}
    
    def validation_step(self, batch, batch_idx):
        loss,logits = self(batch['input_ids'], batch['attention_mask'],batch['labels'])
        
        labels = batch['labels']
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.cfg['num_labels']) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        #active_labels = torch.where(active_accuracy, labels.view(-1), torch.tensor(-100).type_as(labels))
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        #tr_labels.extend(labels)
        #tr_preds.extend(predictions)
        # metric to change
        try:
            score = accuracy_score(labels.cpu().numpy(), predictions.cpu().detach().numpy())
        except ValueError:
            score=0.0
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_score', score, on_step=False, on_epoch=True, prog_bar=True)

        
    
    def configure_optimizers(self):      
        optimizer = AdamW(
            self.parameters(), lr=self.cfg['lr'],
            #weight_decay=self.hparams.weight_decay
            )

        #CosineAnnealingLR(optimizer=opt, eta_min=self.hparams.min_lr, T_max=self.hparams.T_max)
        scheduler = {
            #'scheduler':CyclicLR(optimizer,base_lr=1e-7, max_lr=2e-2,step_size_up=self.hparams.steps_lr//2,mode="triangular2",cycle_momentum=False),
            'scheduler':OneCycleLR(optimizer,
            max_lr=self.cfg['max_lr'],steps_per_epoch=self.cfg['steps_lr'], 
            epochs=self.cfg['epochs'], #pct_start =0.1,
            ),
            'name':self.cfg['scheduler'],
            'interval':'step',
            'frequency': 1
            }
        
        return [optimizer], [scheduler]





if __name__ == '__main__':
    
    CFG = CFG()
    output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

    labels_to_ids = {v:k for k,v in enumerate(output_labels)}
    ids_to_labels = {k:v for k,v in enumerate(output_labels)}

    train_text_df = pd.read_csv("../input/train_NER.csv")
    train_df = pd.read_csv("../input/train.csv")

    # CHOOSE VALIDATION INDEXES
    IDS = train_df.id.unique()
    print('There are',len(IDS),'train texts. We will split 90% 10% for validation.')

    # TRAIN VALID SPLIT 90% 10%
    train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
    
    # CREATE TRAIN SUBSET AND VALID SUBSET
    data = train_text_df[['id','text', 'entities']]
    train_dataset = data.loc[data['id'].isin(IDS[train_idx]),['text', 'entities']].reset_index(drop=True)
    test_dataset = data.loc[data['id'].isin(IDS[valid_idx])].reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))
    
        
        ###
    tag = 'ep{}-'.format(CFG.epochs)

    CFG.steps_lr=len(train_dataset)//(CFG.batch_size) + 1

    tokenizer = AutoTokenizer.from_pretrained(CFG.tokpath,add_prefix_space=True)

    dm = DataModule(train_dataset, test_dataset, tokenizer, None, CFG)
    model = ModelModule(CFG.__dict__)


    filename = f"{CFG.model_name}-{tag}"
    checkpoint_callback = ModelCheckpoint(monitor='val_score', dirpath='./', mode='max', filename=filename,save_top_k=1)
    lr_logger = LearningRateMonitor(logging_interval="step")

    trainer = Trainer(
        gpus=CFG.n_procs,
        max_epochs=CFG.epochs,
        precision=CFG.precision,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback,lr_logger],
        #strategy=CFG.stg,
        #log_every_n_steps=5,
        )

    trainer.fit(model, datamodule=dm)
        
    #del model
    gc.collect()
    torch.cuda.empty_cache()