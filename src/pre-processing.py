import pandas as pd
from tqdm import tqdm
import os

if __name__ == '__main__':
    train_df = pd.read_csv('../input/corrected_train.csv')
    test_names, train_texts = [], []
    for f in tqdm(list(os.listdir('../input/train'))):
        test_names.append(f.replace('.txt', ''))
        train_texts.append(open('../input/train/' + f, 'r').read())
    train_text_df = pd.DataFrame({'id': test_names, 'text': train_texts})
    all_entities = []
    for ii,i in enumerate(train_text_df.iterrows()):
        if ii%100==0: print(ii,', ',end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]: entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities
    train_text_df.to_csv('../input/train_NER.csv',index=False)