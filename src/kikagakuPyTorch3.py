import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Net(pl.LightningModule):
    
    # New: バッチサイズ等を引数に指定
    def __init__(self, input_size=10, hidden_size=5, output_size=3, batch_size=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size
    
    # 変更なし
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    # New: 目的関数の設定
    def lossfun(self, y, t):
        return F.cross_entropy(y, t)
    
    # New: optimizer の設定
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)
    
    # New: train 用の DataLoader の設定
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)
    
    # New: 学習データに対する処理
    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results
    
    # New: val 用の DataLoader の設定
    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()
        results = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return results
    
    # New: テストデータに対する処理
    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size)
    
    # New: テストデータに対するイテレーションごとの処理
    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results
    
    # New: テストデータに対するエポックごとの処理
    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results

if __name__ == '__main__':
    # Wine データセットの読み込み

    df = pd.read_csv('data/wine_class.csv')

    np.unique(df['Class'], return_counts=True)

    # axis=1で列方向に和を計算
    x = df.drop('Class', axis=1)
    t = df['Class']

    # Tensor形式に変換
    x = torch.tensor(x.values, dtype=torch.float32)
    t = torch.tensor(t.values, dtype=torch.int64)
    # 分類の場合、ラベルは0から始まる必要があるため、1を引く
    t = t - 1

    dataset = torch.utils.data.TensorDataset(x, t)
    n_train = int(len(dataset) * 0.6)
    n_val = int(len(dataset) * 0.2)
    n_test = len(dataset) - n_train - n_val

    train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    torch.manual_seed(0)

    net = Net()
    trainer = pl.Trainer()
    trainer.fit(net)

    trainer.test()