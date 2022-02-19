
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

class Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        LABEL_ALL_SUBTOKENS=True
        
        # GET TEXT AND WORD LABELS 
        text = self.data.text[index]        
        word_labels = self.data.entities[index]
        text_id = self.data.id[index]

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
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids: # tokens wise                       
            if word_idx is None:# padding
                label_ids.append(-100) # label ignored in loss
            elif word_idx != previous_word_idx: # change word   
                label_ids.append( labels_to_ids[word_labels[word_idx]] )
            else: # same word
                if LABEL_ALL_SUBTOKENS:
                    label_ids.append( labels_to_ids[word_labels[word_idx]] )
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx
            
        return {
            'input_ids': torch.tensor(encoding['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoding['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'word_ids':str(word_ids),
            'text_id':text_id
        }


class DataModule(LightningDataModule):
    def __init__(self, train_df, val_df, tokenizer, cfg=None):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.cfg = cfg
        self.tokenizer = tokenizer
        
    
    def setup(self,stage):
        if stage == 'fit':
            self.train_ds = Dataset(self.train_df, self.tokenizer, self.cfg.max_length)
            self.valid_ds = Dataset(self.val_df,   self.tokenizer, self.cfg.max_length)

    
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




