from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
# import matplotlib.pyplot as plt
from math import log

epoch = 8

class Sigmoid:
    def __call__(self, X):
        """
        activation function (used for conv2d)
        :param X: input of shape (batch, ...)
        :return Y: output vector of the same shape.
        """
        self.X = X
        Y = 1 / (1 + np.exp(-X))
        return Y
    
    def bp(self, gradient):
        X = self.X
        return gradient * np.exp(-X) / (1 + np.exp(-X))

    def step(self, alpha):
        pass


class Softmax:
    def __call__(self, x):
        """
        Activation function used for fully-connected layer
        x: shape (batch, num_classes)
        returns: shape (batch, num_classes)
        """
        # subtract max for numerical stability
        x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def bp(self, gradient):
        pass

    def step(self, alpha):
        pass


class Flatten:
    def __call__(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    def bp(self, gradient):
        return np.reshape(gradient, self.shape)
    def step(self, alpha):
        pass


class Pooling:
    def __call__(self, X, k=2, stride=2):
        """
        :param X: input of shape (..., height, width)
        """
        self.X_shape = X.shape
        self.k = k
        self.stride = stride
        # Create sliding windows over last two dims
        windows = np.lib.stride_tricks.sliding_window_view(
            X,
            window_shape=(k, k),
            axis=(-2, -1)
        )
        # Apply stride over spatial dimensions
        windows = windows[..., ::stride, ::stride, :, :]
        out_h, out_w = windows.shape[-4], windows.shape[-3]
        self.out_h, self.out_w = out_h, out_w

        # Flatten the k×k window, find argmax, then convert back to 2D indices
        flat = windows.reshape(*windows.shape[:-2], k * k)  # (..., out_h, out_w, k*k)
        self.argmax_flat = flat.argmax(axis=-1)             # (..., out_h, out_w)

        return flat.max(axis=-1).reshape(*X.shape[:-2], out_h, out_w)
    
    def bp(self, gradient):
        k, stride = self.k, self.stride
        H, W = self.X_shape[-2], self.X_shape[-1]
        batch_size = int(np.prod(self.X_shape[:-2]))

        i = np.arange(self.out_h)
        j = np.arange(self.out_w)

        h_idx = (i[:, None] * stride + self.argmax_flat // k)  # (..., out_h, out_w)
        w_idx = (j[None, :] * stride + self.argmax_flat % k)

        flat_idx = (h_idx * W + w_idx).reshape(batch_size, -1)  # (batch, out_h*out_w)
        grad_flat = gradient.reshape(batch_size, -1)             # (batch, out_h*out_w)

        dX_flat = np.array([
            np.bincount(flat_idx[b], weights=grad_flat[b], minlength=H * W)
            for b in range(batch_size)
        ])

        return dX_flat.reshape(self.X_shape)

    def step(self, alpha):
        pass


class CrossEntropyLoss:
    def __init__(self, model):
        self.model = model

    def __call__(self, predicts, targets):
        """
        targets:  one-hot labels, shape (batch, num_classes)
        predicts: probabilities, shape (batch, num_classes)
        """
        self.predicts = predicts
        self.targets = targets
        eps = 1e-12
        predicts = np.clip(predicts, eps, 1.0 - eps)

        loss = -np.mean(np.sum(targets * np.log(predicts), axis=1))
        return loss

    def bp(self):
        predicts, targets, graph = self.predicts, self.targets, self.model.graph
        if type(graph[-1]) is Softmax:
            gradient = predicts - targets
        for layer in reversed(graph[:-1]):
            gradient = layer.bp(gradient)


class FullyConnectedLayer:
    def __init__(self, in_num, out_num):
        """
        :param in_num: input size
        :param out_num: output size
        """
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.rand(in_num, out_num)
        self.B = np.random.rand(out_num)
        pass

    def __call__(self, X):
        """
        :param x: input of shape (in_num)
        :return y: output of shape (out_num)
        """
        self.X = X
        return np.dot(X, self.W) + self.B
    
    def bp(self, gradient):
        self.B_grad = np.mean(gradient, axis=0)
        self.W_grad = np.dot(self.X.T, gradient) / gradient.shape[0]
        return np.dot(gradient, self.W.T)
    
    def step(self, alpha):
        self.B -= self.B_grad * alpha
        self.W -= self.W_grad * alpha


class Conv2dLayer:
    def __init__(self, in_c, out_c, k, **kwargs):
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.kernel = np.random.rand(k * k * in_c, out_c)
        self.bias = np.random.rand(out_c)
        self.stride = kwargs.get("stride", 1)
        self.padding = kwargs.get("padding", 0)

    def _im2col(self, img):
        """Extract all patches: (batch, out_h, out_w, k*k*in_c)"""
        windows = np.lib.stride_tricks.sliding_window_view(
            img, window_shape=(self.k, self.k), axis=(-2, -1)
        )                                                        # (batch, in_c, out_h, out_w, k, k)
        windows = windows[..., ::self.stride, ::self.stride, :, :]
        batch, in_c, out_h, out_w, _, _ = windows.shape
        # Rearrange to (batch, out_h, out_w, k*k*in_c)
        return (windows
                .transpose(0, 2, 3, 1, 4, 5)
                .reshape(batch, out_h, out_w, -1))

    def __call__(self, img):
        in_c, out_c, k, stride, padding = self.in_c, self.out_c, self.k, self.stride, self.padding
        if padding > 0:
            img = np.pad(img, [(0,0), (0,0), (padding,padding), (padding,padding)], mode="constant")
        self.img = img

        self.col = self._im2col(img)                             # (batch, out_h, out_w, k*k*in_c)
        self.out_h, self.out_w = self.col.shape[1], self.col.shape[2]

        out = self.col @ self.kernel + self.bias                 # (batch, out_h, out_w, out_c)
        return out.transpose(0, 3, 1, 2)                         # (batch, out_c, out_h, out_w)

    def bp(self, gradient):
        # gradient: (batch, out_c, out_h, out_w)
        grad = gradient.transpose(0, 2, 3, 1)                   # (batch, out_h, out_w, out_c)

        self.bias_grad = grad.mean(axis=(0, 1, 2))               # (out_c,)

        # col: (batch, out_h, out_w, k*k*in_c)
        col_flat = self.col.reshape(-1, self.k * self.k * self.in_c)   # (batch*out_h*out_w, k*k*in_c)
        grad_flat = grad.reshape(-1, self.out_c)                        # (batch*out_h*out_w, out_c)
        self.kernel_grad = col_flat.T @ grad_flat / col_flat.shape[0]   # (k*k*in_c, out_c)

    def step(self, alpha):
        self.bias -= self.bias_grad * alpha
        self.kernel -= self.kernel_grad * alpha
                

class Optimizer:
    def __init__(self, model, **kwargs):
        self.graph = model.graph
        self.alpha = kwargs.get("alpha", 1e-3)

    def step(self):
        for layer in self.graph:
            layer.step(self.alpha)        


class SimpleNetwork:
    """
    A simple 4-layer neural network consisting of :
        Conv2d
        Pooling
        FullyConnected
        Activation
    """
    def __init__(self):
        self.graph = [
            Conv2dLayer(1, 10, 5, padding=1),
            Sigmoid(),
            Pooling(),
            Flatten(),
            FullyConnectedLayer(13 * 13 * 10, 10),
            Softmax()
        ]
        self.cnn1 = Conv2dLayer(1, 10, 5, padding=1)
        self.fc1 = FullyConnectedLayer(13 * 13 * 10, 10)
    
    def __call__(self, x):
        """
        :params: x: input image. (batch, channel, height, weight)
        :return: predict: one-hot vector.
        """
        for calculate in self.graph:
            x = calculate(x)
        return x


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
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
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

    for i in range(1, epoch+1):
        mean_loss = 0
        mean_acc = 0
        for images, labels in train_loader:
            images = images.to("cpu").numpy()
            labels = labels.to("cpu").numpy()
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
            images = images.to("cpu").numpy()
            labels = labels.to("cpu").numpy()
            targets = np.eye(10)[labels]

            predicts = model(images)
            loss = lossfunc(predicts, targets)

            acc = np.mean(np.argmax(predicts, axis=1) == labels)

            mean_loss += loss
            mean_acc += acc
        mean_loss = round(mean_loss / len(test_loader), 3)
        mean_acc = round(mean_acc * 100 / len(test_loader), 2)
        print(f"test loss: {mean_loss}, testset accuracy: {mean_acc}%")


if __name__ == "__main__":
    main()

