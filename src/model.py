
from torch.optim.lr_scheduler import OneCycleLR

import torch
from pytorch_lightning import LightningModule

# transformer
from transformers import AdamW, AutoConfig,AutoModelForTokenClassification

from sklearn.metrics import accuracy_score


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