
from torch.optim.lr_scheduler import OneCycleLR

import torch
from pytorch_lightning import LightningModule
from metric import * 
import pandas as pd
from ast import literal_eval

# transformer
from transformers import AdamW, AutoConfig,AutoModelForTokenClassification
from deepspeed.ops.adam import FusedAdam
from sklearn.metrics import accuracy_score



output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

class ModelModule(LightningModule):
    def __init__(self, cfg,train_gt,test_gt):
        super().__init__()
        self.df_pred = pd.DataFrame(columns = ['id','discourse_type','predictionstring'])
        self.df_gt_train = train_gt
        self.df_gt_val = test_gt
        
        self.cfg=cfg
        #self.save_hyperparameters(cfg)
        config = AutoConfig.from_pretrained(self.cfg['modelpath'])
        config.num_labels = self.cfg['num_labels']
        self.model = AutoModelForTokenClassification.from_pretrained(self.cfg['modelpath'],config=config)
        #self.model = AutoModel.from_pretrained(self.hparams.modelpath,config=config)
       
        #self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask,labels):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,labels=labels)
        return out.loss,out.logits
    
    def training_step(self, batch, batch_idx):
        loss,logits = self(batch['input_ids'], batch['attention_mask'],batch['labels']) #(N,seq,labels)    
        pred = torch.argmax(logits, axis=2).cpu().detach().numpy() #(N,seq)
        labels = batch['labels'].cpu().detach().numpy() # (N,seq)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            'loss': loss,
            'pred':pred,
            'word_ids':batch['word_ids'],#(N,seq)
            'text_id':batch['text_id'], # (N)
            'labels':labels
        }
    
    def validation_step(self, batch, batch_idx):
        loss,logits = self(batch['input_ids'], batch['attention_mask'],batch['labels'])
        pred = torch.argmax(logits, axis=2).cpu().detach().numpy() #(N,seq)
        labels = batch['labels'].cpu().detach().numpy() # (N,seq)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return {
            'loss': loss,
            'pred':pred,#(N,seq)
            'word_ids':batch['word_ids'],#(N,seq)
            'text_id':batch['text_id'], #(N)
            'labels':labels #(N,seq)
        }
    
    def training_epoch_end(self, output): # [dict1,dict2]:epoch,batch
        if not self.df_gt_train.empty:
            self.gt_trTrue=True
        
        for bat in output: # pred, word_ids, text_id,label
            pred = bat['pred']
            word_ids = [literal_eval(e) for e in bat['word_ids']]
            text_id  = bat['text_id']
            labels = bat['labels']
            
            for i in range(len(text_id)):
                self.build_df(text_id[i],pred[i],labels[i],word_ids[i])
        
        f1 = score_feedback_comp3(self.df_pred,self.df_gt_train)
        #self.logger.experiment.add_scalar('train_f1', f1, global_step=self.current_epoch)
        self.log('train_f1', f1,prog_bar=True)
        
        self.df_pred = pd.DataFrame(columns = ['id','discourse_type','predictionstring'])
        
    
    def validation_epoch_end(self,output):
        if not self.df_gt_val.empty:
            self.gt_valTrue=True
        
        for bat in output:
            pred = bat['pred']
            word_ids = [literal_eval(e) for e in bat['word_ids']]
            text_id  = bat['text_id']
            labels = bat['labels']
            
            for i in range(len(text_id)):
                self.build_df(text_id[i],pred[i],labels[i],word_ids[i])
        
        f1 = score_feedback_comp3(self.df_pred,self.df_gt_val)
        self.log('val_f1', f1,prog_bar=True)
        
        self.df_pred = pd.DataFrame(columns = ['id','discourse_type','predictionstring'])
    
    
    def build_df(self,id_, pred_, labels_, word_ids_):
        #text_id,pred,labels,word_ids #self.df_pred #self.df_gt
        text_allpreds = []
        
        pred_ = [ids_to_labels[i] if i!=-100 else 'PAD' for i in pred_] # token wise
        #labels_ = [ids_to_labels[i] if i!=-100 else 'PAD' for i in labels_ ] # token wise

        prediction = [] #word wise
        #gt = []
        
        previous_word_idx = -1
        for idx,word_idx in enumerate(word_ids_):                            
            if word_idx!=None and word_idx != previous_word_idx:
                # use only first subword pred  
                prediction.append(pred_[idx])
                #if not (self.gt_valTrue and self.gt_trTrue):
                    #gt.append(labels_[idx])
                previous_word_idx = word_idx
        
        j = 0
        end = 0
        while j < len(prediction):
            if prediction[j]=='O':
                j+=1
            else:
                cls = prediction[j].replace('B','I') # Take I and B
                end = j + 1
                while end < len(prediction) and prediction[end] == cls:
                    end += 1
                
                if end - j > 7: # 7 to check
                    self.df_pred = self.df_pred.append(
                        pd.Series([id_, cls.replace('I-','') ,' '.join(map(str, list(range(j, end))))], index = self.df_pred.columns), 
                        ignore_index=True)
                j = end
         
        
    def configure_optimizers(self):      
        optimizer = FusedAdam(
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
        
        