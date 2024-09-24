
import transformers
import torch
import torchmetrics
import pytorch_lightning as pl
from model.loss import get_loss
from model.optimizer import get_optimizer
from model.MultiTaskLoss import MultiTaskLoss

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config["model"]["name"]
        self.lr = config["training"]["learning_rate"]

        # 사용할 모델을 호출합니다.
        self.multi_task = config["model"]["multi_tasks"]
        if not self.multi_task:
            self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path=self.model_name, num_labels=1)
        else:
            self.plm = transformers.AutoModel.from_pretrained(pretrained_model_name_or_path=self.model_name)
            self.classifier1 = torch.nn.Linear(self.plm.config.hidden_size, 1)
            self.classifier2 = torch.nn.Linear(self.plm.config.hidden_size, 1)
            
        # Loss 계산을 위해 사용될 L1Loss를 호출합니다.
        self.loss_func = get_loss(config["training"]["loss"])
        if self.multi_task:
            self.loss_func = MultiTaskLoss(self.loss_func)
        self.optimizer = get_optimizer(config["training"]["optimizer"])

    def forward(self, x):
        if not self.multi_task:
            if not isinstance(x, dict):
                raise ValueError("Invalid input format. Expected a dictionary.") # 디버깅
            outputs = self.plm(**x)
            logits = outputs['logits']
            if self.training:
                return logits
            logits = torch.clamp(logits, min=0, max=5)
            return logits
        else:
            outputs = self.plm(**x)
            cls_token = outputs.last_hidden_state[:, 0, :]
            
            logits1 = self.classifier1(cls_token) # logits1: regression task
            logits2 = self.classifier2(cls_token) # logits2: binary classification task
            logits_all = torch.cat([logits1, logits2], dim=1)
            return logits_all
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y.float())
        self.log("val_loss", loss)

        # prog_bar=True를 통해 epoch마다 progress bar에 val_pearson 점수 출력
        if self.multi_task:
            self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits[:, 0].squeeze(), y[:, 0].squeeze()), prog_bar=True)
        else:
            self.log("val_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()), prog_bar=True)


    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        self.log("test_pearson", torchmetrics.functional.pearson_corrcoef(logits.squeeze(), y.squeeze()))

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

