import argparse

import pytorch_lightning as pl
import torch
import pandas as pd

from utils.config import load_config
from data_pipeline.dataloader import Dataloader
from model.model import Model
 
if __name__ == '__main__':
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or inference mode')
    args = parser.parse_args(args=[])

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(config)

    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=config["training"]["epochs"], log_every_n_steps=1)
    
    if args.mode == 'train':
        model = Model(config)   
        # Train part
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        # 학습이 완료된 모델을 저장합니다.
        torch.save(model, 'model.pt')
    elif args.mode == 'inference':
        # Inference part
        # 저장된 모델로 예측을 진행합니다.
        model = torch.load('model.pt')
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))

        # output 형식을 불러와서 예측된 결과로 바꿔주고, output.csv로 출력합니다.
        output = pd.read_csv('../data/sample_submission.csv')
        output['target'] = predictions
        output.to_csv('output.csv', index=False)
    else:
        raise ValueError('mode should be either "train" or "inference"')
