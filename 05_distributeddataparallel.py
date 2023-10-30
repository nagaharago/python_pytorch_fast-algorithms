import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from utils import Net
import GPUtil


def train(rank, world_size):
    # 各プロセスのセットアップ
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.manual_seed(0)

    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        sampler=train_sampler,
    )

    # モデルとオプティマイザの設定
    model = Net().cuda(rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(10):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(rank), target.cuda(rank)
            optimizer.zero_grad()
            outputs = ddp_model(data)
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
                    f"[{rank}] Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
                )

    # 終了
    dist.destroy_process_group()


def main():
    world_size = 2  # 2つのプロセスを起動します。適宜調整してください。
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
