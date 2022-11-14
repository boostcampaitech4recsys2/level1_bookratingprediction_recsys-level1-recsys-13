import tqdm
import pandas as pd
import torch
import wandb
import os

from ._models import _FactorizationMachineModel, _FieldAwareFactorizationMachineModel
from ._models import rmse, RMSELoss



def predicts_map(x: float) -> float:
    if x < 1:
        return 1.0
    elif x > 10:
        return 10.0
    else:
        return x

class FactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.model_name = args.MODEL
        self.args = args

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.test_dataloader = data['test_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.data_path = args.DATA_PATH
        self.batch_size = args.BATCH_SIZE

        self.device = args.DEVICE

        self.model = _FactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        self.wandb_model_name = args.MODEL
        self.wandb_mode = args.WANDB


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        best_rmse_score = 9999
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                self.model.zero_grad()
                fields, target = fields.to(self.device), target.to(self.device)

                y = self.model(fields)
                loss = self.criterion(y, target.float())

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            if self.wandb_mode :
                wandb.log({"FM RMSE": rmse_score})
            if rmse_score < best_rmse_score :
                best_rmse_score = rmse_score
                torch.save(self.model, os.path.join('models', f"{self.model_name}_model.pt"))
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
        print('best rmse:', best_rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.load_state_dict(torch.load(os.path.join('models', f"{self.model_name}_model.pt"))['state'])
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        submission = pd.read_csv(self.data_path + 'sample_submission.csv')
        submission['rating'] = predicts
        submission['rating'] = submission['rating'].apply(predicts_map)
        submission.to_csv(f'submit/{self.wandb_model_name}_EMBED_DIM{self.embed_dim}_EPOCHS{self.epochs}_WD{self.weight_decay}_DATA_PATH_{self.data_path[5:-1]}_BATHC_SIZE{self.batch_size}.csv', index=False)

        return predicts


class FieldAwareFactorizationMachineModel:

    def __init__(self, args, data):
        super().__init__()

        self.criterion = RMSELoss()

        self.model_name = args.MODEL
        self.args = args

        self.train_dataloader = data['train_dataloader']
        self.valid_dataloader = data['valid_dataloader']
        self.test_dataloader = data['test_dataloader']
        self.field_dims = data['field_dims']

        self.embed_dim = args.FFM_EMBED_DIM
        self.epochs = args.EPOCHS
        self.learning_rate = args.LR
        self.weight_decay = args.WEIGHT_DECAY
        self.log_interval = 100
        self.data_path = args.DATA_PATH
        self.batch_size = args.BATCH_SIZE

        self.device = args.DEVICE

        self.model = _FieldAwareFactorizationMachineModel(self.field_dims, self.embed_dim).to(self.device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, amsgrad=True, weight_decay=self.weight_decay)

        self.wandb_model_name = args.MODEL
        self.wandb_mode = args.WANDB


    def train(self):
      # model: type, optimizer: torch.optim, train_dataloader: DataLoader, criterion: torch.nn, device: str, log_interval: int=100
        best_rmse_score = 9999
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            tk0 = tqdm.tqdm(self.train_dataloader, smoothing=0, mininterval=1.0)
            for i, (fields, target) in enumerate(tk0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                if (i + 1) % self.log_interval == 0:
                    tk0.set_postfix(loss=total_loss / self.log_interval)
                    total_loss = 0

            rmse_score = self.predict_train()
            if self.wandb_mode :
                wandb.log({"FFM RMSE": rmse_score})
            if rmse_score < best_rmse_score :
                best_rmse_score = rmse_score
                torch.save(self.model, os.path.join('models', f"{self.model_name}_model.pt"))
            print('epoch:', epoch, 'validation: rmse:', rmse_score)
        print('best rmse:', best_rmse_score)


    def predict_train(self):
        self.model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            for fields, target in tqdm.tqdm(self.valid_dataloader, smoothing=0, mininterval=1.0):
                fields, target = fields.to(self.device), target.to(self.device)
                y = self.model(fields)
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
        return rmse(targets, predicts)


    def predict(self, dataloader):
        self.model.load_state_dict(torch.load(os.path.join('models', f"{self.model_name}_model.pt"))['state'])
        self.model.eval()
        predicts = list()
        with torch.no_grad():
            for fields in tqdm.tqdm(dataloader, smoothing=0, mininterval=1.0):
                fields = fields[0].to(self.device)
                y = self.model(fields)
                predicts.extend(y.tolist())
        submission = pd.read_csv(self.data_path + 'sample_submission.csv')
        submission['rating'] = predicts
        submission['rating'] = submission['rating'].apply(predicts_map)
        submission.to_csv(f'submit/{self.wandb_model_name}_EMBED_DIM{self.embed_dim}_EPOCHS{self.epochs}_WD{self.weight_decay}_DATA_PATH_{self.data_path[5:-1]}_BATHC_SIZE{self.batch_size}.csv', index=False)

        return predicts
