import pandas as pd
import random

from tqdm.auto import tqdm

import transformers
import torch
import pytorch_lightning as pl

from data_pipeline.dataset import Dataset
from data_pipeline.augmentation import Augmentation
import re

class Dataloader(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.model_name = config["model"]["name"]
        self.batch_size = config["training"]["batch_size"]
        self.shuffle = config["training"]["shuffle"]

        self.train_path = config["path"]["train"]
        self.dev_path = config["path"]["dev"]
        self.test_path = config["path"]["test"]
        self.predict_path = config["path"]["predict"]

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name, 
                                                                    clean_up_tokenization_spaces = True, # 향후 버전에서 기본값 False임.(토큰화된 문자열을 원래 문자열로 더 정확하게 복원하기 위해 공백을 살리는 취지)
                                                                    max_length=config["data"]["max_tokens"])
        self.target_columns = ['label']
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']
        # self.augmentation_method = config["data"].get("augmentation", {}).get("method", None)
        # self.augmentation_probability = config["data"].get("augmentation", {}).get("probability", 0.0)
        self.augmentation = Augmentation(config)

    def augment_data(self, dataframe):
        # if not self.augmentation_method or self.augmentation_probability <= 0:
        #     return dataframe

        # augmented_data = []
        # for idx, item in tqdm(dataframe.iterrows(), desc='augmenting', total=len(dataframe)):
        #     # augmentation_probability의 확률로 데이터를 증강
        #     if random.random() > self.augmentation_probability or item['label'] == 0:
        #         continue
            
        #     augmented_item = item.copy()
        #     if self.augmentation_method == "swap_sentences":
        #         augmented_item[self.text_columns[0]], augmented_item[self.text_columns[1]] = (
        #             augmented_item[self.text_columns[1]], augmented_item[self.text_columns[0]]
        #         )
        #     augmented_data.append(augmented_item)
        
        # augmented_dataframe = pd.DataFrame(augmented_data)
        # return pd.concat([dataframe, augmented_dataframe], ignore_index=True)
        return self.augmentation(dataframe)
    
    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = '[SEP]'.join([item[text_column] for text_column in self.text_columns])
            outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
            
            # 패딩 부분을 구분할 수 있도록 attention_mask 살려서 data에 넣기
            data.append({'input_ids':outputs['input_ids'],
                         'attention_mask': outputs['attention_mask']
                         })
        return data

    def cleaning(self, dataframe):
        # 띄어쓰기 교정을 위한 spacing 객체를 생성합니다.

        for idx, item in tqdm(dataframe.iterrows(), desc='cleaning', total=len(dataframe)):
            for text_column in self.text_columns:
                # 정규 표현식을 사용하여 3회 이상 반복되는 문자를 2회로 줄이고, 띄어쓰기 교정을 진행합니다.
                item[text_column] = re.sub(r'(.)\1{2,}', r'\1\1', item[text_column]) if isinstance(item[text_column], str) else item[text_column]
                item[text_column] = ' '.join(self.tokenizer.tokenize(item[text_column]))
                
                dataframe.loc[idx, text_column] = item[text_column]

        return dataframe

    def preprocessing(self, data):
        # 안쓰는 컬럼을 삭제합니다.
        data = data.drop(columns=self.delete_columns)

        # 텍스트 데이터 클리닝을 진행합니다.
        data = self.cleaning(data)

        # 타겟 데이터가 없으면 빈 배열을 리턴합니다.
        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터와 검증 데이터셋을 호출합니다
            train_data = pd.read_csv(self.train_path)
            train_data = self.augment_data(train_data)
            val_data = pd.read_csv(self.dev_path)

            # 학습데이터 준비
            train_inputs, train_targets = self.preprocessing(train_data)

            # 검증데이터 준비
            val_inputs, val_targets = self.preprocessing(val_data)

            # train 데이터만 shuffle을 적용해줍니다, 필요하다면 val, test 데이터에도 shuffle을 적용할 수 있습니다
            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            # 평가데이터 준비
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, predict_targets = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs, [])

    def train_dataloader(self):
        # cpu 워커 좀 더 써보자... 지금 기본값이라 메인 프로세스만 쓰는중
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers = 2)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers = 2)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers = 2)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.predict_dataset, batch_size=self.batch_size, num_workers = 2)