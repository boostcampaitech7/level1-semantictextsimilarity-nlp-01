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
    merged_config = base_config.copy()
    for key, value in wandb_config.items():
        if isinstance(value, dict) and key in merged_config:
            merged_config[key] = load_and_merge_config(merged_config[key], value)
        else:
            keys = key.split('.')
            current = merged_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value
    return merged_config


def get_trainer(config, log_name):
    return pl.Trainer(
        accelerator ="gpu",
        devices=1,
        max_epochs=config["training"]["epochs"],
        log_every_n_steps=1,
        logger =[CSVLogger(save_dir="logs", name=log_name),
                 pl.loggers.WandbLogger()]
                    )
 
 
def train(base_config, model_path):
    
    # wandb 설정. wandb.init()을 호출하면 자동으로 sweep설정을 가져옴.
    with wandb.init(project="U-4-do") as run:
        config = load_and_merge_config(base_config, dict(wandb.config))
        # dataloader와 model을 생성
        dataloader = Dataloader(config)
        
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = Model(config)
            model = torch.load(model_path)
            
        else:
            print("Creating new model")
            model = Model(config)
    
        # logging name 설정
        log_name = f"{config['model']['name']}_bs{config['training']['batch_size']}_lr{config['training']['learning_rate']}"
    
        # trainer 설정
        trainer = get_trainer(config, log_name)
        
        # 학습 및 테스트
        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)
        
        
        
        if not os.path.exists(model_path):
            torch.save(model, model_path)
        else:
            torch.save(model, f"model_{wandb.run.id}.pt")
        

def main():
    base_config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train or inference mode')
    parser.add_argument('--model_path', type=str, default ='model.pt', help='path to the model weights file')
    parser.add_argument('--sweep', type=bool, default=False, help='whether to use wandb sweep')
    
    args = parser.parse_args()
        
    # 재현을 위한 seed 고정
    set_seed(base_config["seed"])
    
    secrets = load_config(config_path="secrets.yaml")
    wandb.login(key=secrets["wandb-api-key"])

    if args.mode == 'train':
        print("Running on train mode")
        
        if args.sweep:
            print("Running Sweep")
            sweep_config = {
                            'method': 'bayes',
                            'metric': {'name': 'val_loss', 'goal': 'minimize'},
                            'parameters': {
                                'training.epochs': {'values': [5, 8, 15, 20]},
                                'training.batch_size': {'values': [8, 16, 32]},
                                'training.learning_rate': {'min': 1e-5, 'max': 5e-4},
                                'training.optimizer': {'values': ['AdamW']},
                                'training.loss': {'values': ['L1loss', 'MSEloss', 'HuberLoss']},
                                'data.dropout_rate': {'min': 0.1, 'max': 0.2},
                                'data.augmentation_probability': {'min': 0.4, 'max': 1.0}
                            }
                            }
        
            sweep_id = wandb.sweep(sweep_config, project="U-4-do", entity='nlp-01')
            wandb.agent(sweep_id, lambda: train(base_config, args.model_path), count = 5) # 5번 실험 실행
        
        else:
            print("Running without wandb sweep")
            train(base_config, args.model_path)
            
       
    elif args.mode == 'inference':
        
        # 디버깅 코드 추가 mode에 inference 입력시 이하 코드 출력
        print('Running on Inference Mode')
        config = load_config()
        dataloader = Dataloader(config)
        
        # 저장된 모델로 예측을 진행
        model = torch.load(args.model_path)
        trainer = get_trainer(config, "inference")
        predictions = trainer.predict(model=model, datamodule=dataloader)

        # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
        predictions = list(round(float(i), 1) for i in torch.cat(predictions))
        
        output_format = pd.read_csv('./data/sample_submission.csv')
        output = pd.DataFrame(columns=output_format.columns) # output 형식의 columns만 참고
        predict_csv = pd.read_csv(config["path"]["predict"])
        for col in output.columns:
            if col in predict_csv.columns:
                output[col] = predict_csv[col] # output의 비어있는 행들은 predict_csv의 값으로 채워줌
        output['target'] = predictions
        if not os.path.exists('outputs'):
            os.makedirs('outputs')
        output.to_csv('outputs/output.csv', index=False)
    else:
        raise ValueError('mode should be either "train" or "inference"')

if __name__ == '__main__':
    main()
