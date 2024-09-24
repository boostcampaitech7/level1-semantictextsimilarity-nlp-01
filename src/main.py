import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
import torch
import pandas as pd
import random
import wandb

from utils.config import load_config
from data_pipeline.dataloader import Dataloader
from model.model import Model
import os


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # CPU 시드 고정
    torch.cuda.manual_seed(seed)  # GPU 시드 고정
    torch.cuda.manual_seed_all(seed)  # 모든 GPU에 대한 시드 고정 (멀티 GPU 환경)

def load_and_merge_config(base_config, wandb_config):
    return {**base_config, **wandb_config}

def get_trainer(config, log_name):
    return pl.Trainer(
        accelerator ="gpu",
        devices=1,
        max_epochs=config["training"]["epochs"],
        log_every_n_steps=1,
        logger =[CSVLogger(save_dir="logs", name=log_name),
                 pl.loggers.WandbLogger()]
                    )
 
def train(base_config):
    
    # wandb 설정
    with wandb.init(config = base_config) as run:
        config = load_and_merge_config(base_config, run.config)
        # dataloader와 model을 생성
        dataloader = Dataloader(config)
        
        if model_path and os.path.exists(model_path):
            print("저장된 모델을 확인하였으며 해당 모델을 load하여 학습합니다")
            print(f"Loading model from {model_path}")
            model = torch.load(model_path)
        else:
            print("model_path를 확인하지 못하였으며 config.yaml에 있는 모델명으로 새로 학습합니다")
            print("Creating new model")
            model = Model(config)
        
        model = Model(config)
        
        # logging name 설정
        log_name = f"{config['model']['name']}_bs{config['training']['batch_size']}_lr{config['training']['learning_rate']}"
    
        # trainer 설정
        trainer = get_trainer(config, log_name)
        
        # 학습 및 테스트
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        
        # 학습이 완료된 모델을 저장함.
        torch.save(model, f"model_{wandb.run.id}.pt")
        

def main():
    base_config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or inference mode')
    parser.add_argument('--model_path', type=str, default ='model.pt', help='path to the model weights file')
    args = parser.parse_args()
        
    # 재현을 위한 seed 고정
    set_seed(base_config["seed"])
    
    if args.mode == 'train':
        print("Running on train mode")
        
        sweep_config = {
            'method': 'random',
            'metric': {'name': 'val_loss', 'goal': 'minimize'},
            'parameters': {
                'training.batch_size': {'values': [4, 8, 16]},
                'training.learning_rate': {'min': 1e-5, 'max': 1e-3},
                'training.epochs': {'values': [3, 5, 10, 20]},
                'training.weight_decay': {'min': 0.001, 'max': 0.1},
                'training.optimizer': {'values': ['AdamW', 'Adam', 'SGD']},
                'training.scheduler': {'values': ['CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau']},
                'training.loss': {'values': ['L1Loss', 'L2Loss']},
                'data.dropout_rate': {'min': 0.1, 'max': 0.5},
                'data.augmentation.0.params.probability': {'min': 0.5, 'max': 1.0}
            }
        }
        
        wandb.login(key="aadeea1600199a35fa358306a6e7c09a4240f709")
        sweep_id = wandb.sweep(sweep_config, project="U-4-do", entity='nlp-01')
        wandb.agent(sweep_id, lambda: train(base_config), count = 5) # 5번 실험 실행
       
    elif args.mode == 'inference':
        
        # 디버깅 코드 추가 agr에 inference 입력시 이하 코드 출력
        print('Running on Inference Mode')
        config = load_config()
        dataloader = Dataloader(config)
        
        # 저장된 모델로 예측을 진행
        model = torch.load(args.model_path)
        trainer = get_trainer(config, "inference")
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

if __name__ == '__main__':
    main()
