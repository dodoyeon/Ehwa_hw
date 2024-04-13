# Pytorch train code
import torch
import torch.nn as nn
import argparse

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torch.optim as optim
from torch_alexnet import AlexNet
import tqdm
import torchvision
import torchvision.transforms as transforms

# import torchsummary


def define_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=3)
    parser.add_argument("batch_size", default=4)
    parser.add_argument("--seed", default=42)
    parser.add_argument("--lr", default=0.001)
    parser.add_argument("--num_class", default=1000)
    parser.add_argument("--eval_step", default=2000)
    parser.add_argument("--save_dir", default="torch_alex")
    args = parser.parse_args()
    return args


def epoch_train(
    args, epoch, model, trainloader, testloader, criterion, optimizer, lr_scheduler
):
    tr_loss = 0
    for step, batch in enumerate(trainloader):
        batch.to(device)
        inputs, targets = batch

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        tr_loss += loss.item()

    te_loss = epoch_test(args, model, testloader, criterion)
    print(
        f"{epoch}epoch - {step}steps : tr loss={tr_loss/step:.3f} | te loss={te_loss/step:.3f}"
    )
    tr_loss = 0

    print(f"-- {epoch}th epoch training done")


def epoch_test(args, model, testloader, criterion):
    model.eval()
    with torch.no_grad():
        te_loss = 0
        for i, batch in enumerate(testloader):
            batch.to(device)
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            te_loss += loss.item()

    # torch.save()


def train(args, modeltrainloader, testloader, criterion, optimizer, lr_scheduler):
    model.train()

    for epoch in tqdm(range(args.epochs)):
        epoch_train(
            args,
            epoch,
            model,
            trainloader,
            testloader,
            criterion,
            optimizer,
            lr_scheduler,
        )

    torch.save(model.state_dict(), args.save_dir)
    print("-- saving model done")
    print("-- training done")


args = define_argument()
model = AlexNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

optimizer = optim.Adam(model.parameters())
lr_scheduler = None
criterion = nn.CrossEntropyLoss()
# torchsummary.summary(model, input_size=(3, 227, 227), device="cuda")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=args.batch_size,
    shuffle=True,
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=args.batch_size,
    shuffle=True,
)

train()
