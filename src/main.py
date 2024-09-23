import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
import pandas as pd
import random

from utils.config import load_config
from data_pipeline.dataloader import Dataloader
from model.model import Model

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # CPU 시드 고정
    torch.cuda.manual_seed(seed)  # GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정 (멀티 GPU 환경)
 
if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or inference mode')
    
    args = parser.parse_args()
    # args = parser.parse_args(args=[])

    # 재현을 위한 seed 고정
    SEED = config["seed"]
    set_seed(SEED)

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config)

    # logging name 설정
    model_name = config["model"]["name"]
    batch_size = config["training"]["batch_size"]
    learning_rate = config["training"]["learning_rate"]
    log_name = f"{model_name}_bs{batch_size}_lr{learning_rate}"

    # CSVLogger 설정
    logger = CSVLogger(save_dir="logs", name=log_name)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(logger=logger, accelerator="gpu", devices=1, max_epochs=config["training"]["epochs"], log_every_n_steps=1)
    
    if args.mode == 'train':
        
        # 디버깅 코드 추가 arg에 train 입력시 이하 코드 출력
        print('Running on train mode')
        model = Model(config)   
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, 'model.pt')
    elif args.mode == 'inference':
        
        # 디버깅 코드 추가 agr에 inference 입력시 이하 코드 출력
        print('Running on Inference Mode')
        
        # Inference part
        # 저장된 모델로 예측을 진행합니다.
        model = torch.load('model.pt')
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        # output = pd.read_csv('./data/sample_submission.csv')
        # output['target'] = predictions
        
        output_format = pd.read_csv('./data/sample_submission.csv')
        output = pd.DataFrame(columns=output_format.columns) # output 형식의 columns만 참고
        predict_csv = pd.read_csv(config["path"]["predict"])
        for col in output.columns:
            if col in predict_csv.columns:
                output[col] = predict_csv[col] # output의 비어있는 행들은 predict_csv의 값으로 채워줌
        output['target'] = predictions
        output.to_csv('output.csv', index=False)
    else:
        raise ValueError('mode should be either "train" or "inference"')
