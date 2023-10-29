from sklearn.datasets import load_iris
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # Iris データセットの読み込み
    x, t = load_iris(return_X_y=True)
    x = torch.tensor(x, dtype=torch.float32)
    t = torch.tensor(t, dtype=torch.int64)

    dataset = torch.utils.data.TensorDataset(x, t)


    n_train = int(len(dataset) * 0.6)
    n_val = int(len(dataset) * 0.2)
    n_test = len(dataset) - n_train - n_val


    torch.manual_seed(0)

    # データセットを分割する
    train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    # バッチサイズを指定したデータローダーを作成する
    batch_size = 10

    # shuffleはデフォルトでFalseのため、学習データのみTrueにする
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    # エポックの数
    max_epoch = 1

    # GPUの選択
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # モデルの定義
    torch.manual_seed(0)
    net = Net().to(device)

    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    for epoch in range(max_epoch):

        for batch in train_loader:

            # バッチサイズ分のデータを取り出す
            x, t = batch

            # 学習時に使用するデバイスへ転送する
            x = x.to(device)
            t = t.to(device)

            # パラメータの勾配を初期化
            optimizer.zero_grad()

            # 予測値の計算
            y = net(x)

            # 目的関数の計算
            loss = criterion(y, t)

            # 正解率の計算
            # dim=1 で行ごとの最大値に対する要素番号を取得（dim=0 は列ごと）
            y_label = torch.argmax(y, dim=1)
            acc = torch.sum(y_label == t) * 1.0 / len(t)
            print('accuracy:', acc)
            # 目的関数を値を表示して確認
            print('loss:', loss.item())

            # 各パラメータの勾配を計算
            loss.backward()

            # パラメータの更新
            optimizer.step()