from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np
from model import SimpleNetwork, CrossEntropyLoss, Optimizer

epoch = 8

def mnist_numpy_collate(batch):
    images, labels = zip(*batch)

    # images: tuple of torch tensors, each shape (1, 28, 28)
    X = torch.stack(images, dim=0).numpy().astype(np.float32, copy=False)
    # X shape: (batch, 1, 28, 28)

    labels = np.array(labels, dtype=np.int64)

    return X, labels


def get_mnist_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """
    Return MNIST train and test DataLoaders. 

    Images shape per batch:
        (batch_size, 1, 28, 28)

    Labels shape per batch:
        (batch_size,)
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=mnist_numpy_collate,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mnist_numpy_collate,
    )

    return train_loader, test_loader


def main() -> None:
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    images, labels = next(iter(train_loader))

    print("Train batch images:", images.shape)
    print("Train batch labels:", labels.shape)
    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    model = SimpleNetwork()
    lossfunc = CrossEntropyLoss(model)
    opt = Optimizer(model, alpha=1e-3)
    best_acc = 0

    for i in range(1, epoch+1):
        mean_loss = 0
        mean_acc = 0
        for images, labels in train_loader:
            # images = images.to("cpu").numpy().astype(np.float32)
            # labels = labels.to("cpu").numpy()
            targets = np.eye(10)[labels]

            predicts = model(images)
            loss = lossfunc(predicts, targets)
            lossfunc.bp()
            opt.step()

            acc = np.mean(np.argmax(predicts, axis=1) == labels)

            mean_loss += loss
            mean_acc += acc
        mean_loss = round(mean_loss / len(train_loader), 3)
        mean_acc = round(mean_acc * 100 / len(train_loader), 2)
        print(f"epoch {i} | train loss: {mean_loss}, trainset accuracy: {mean_acc}%", end=" | ")

        mean_loss = 0
        mean_acc = 0
        for images, labels in test_loader:
            targets = np.eye(10)[labels]

            predicts = model(images)
            loss = lossfunc(predicts, targets)

            acc = np.mean(np.argmax(predicts, axis=1) == labels)

            mean_loss += loss
            mean_acc += acc
        mean_loss = round(mean_loss / len(test_loader), 3)
        mean_acc = round(mean_acc * 100 / len(test_loader), 2)
        print(f"test loss: {mean_loss}, testset accuracy: {mean_acc}%")
        if mean_acc > best_acc:
            model.save("mnist_model.pkl")


if __name__ == "__main__":
    main()

