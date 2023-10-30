import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn.parallel import DataParallel
import GPUtil


def train():
    torch.manual_seed(0)

    # データローダーの設定
    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2
    )

    # モデルとオプティマイザの設定
    model = Net().cuda()
    model = DataParallel(model)  # DataParallelでモデルをラップ
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # トレーニングループ
    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            outputs = model(data)
            loss = F.cross_entropy(outputs, target)
            loss.backward()
            optimizer.step()

            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                print(
                    f"GPU {gpu.id}: {gpu.load*100:.1f}% load, {gpu.memoryUsed}/{gpu.memoryTotal}MB used"
                )

            if batch_idx % 10 == 0:
                print(
                    f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )


if __name__ == "__main__":
    train()
