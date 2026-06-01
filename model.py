import numpy as np
import pickle

class Sigmoid:
    def __call__(self, X):
        """
        activation function (used for conv2d)
        :param X: input of shape (batch, ...)
        :return Y: output vector of the same shape.
        """
        self.Y = 1 / (1 + np.exp(-X))
        return self.Y
    
    def bp(self, gradient, **kwargs):
        return gradient * self.Y * (1 - self.Y)

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

    def bp(self, gradient, **kwargs):
        pass

    def step(self, alpha):
        pass


class Flatten:
    def __call__(self, X):
        self.shape = X.shape
        return X.reshape(X.shape[0], -1)
    def bp(self, gradient, **kwargs):
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
    
    def bp(self, gradient, **kwargs):
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
            layers = graph[:-1]
        else:
            gradient = -targets / predicts
            layers = graph
        
        for i, layer in enumerate(reversed(layers)):
            is_last = True if i==len(layers)-1 else False
            gradient = layer.bp(gradient, backward=not is_last)


class FullyConnectedLayer:
    def __init__(self, in_num, out_num):
        """
        :param in_num: input size
        :param out_num: output size
        """
        self.in_num = in_num
        self.out_num = out_num
        limit = np.sqrt(6 / (in_num + out_num))
        self.W = np.random.uniform(-limit, limit, (in_num, out_num)).astype(np.float32)
        self.B = np.zeros(out_num, dtype=np.float32)
        pass

    def __call__(self, X):
        """
        :param x: input of shape (in_num)
        :return y: output of shape (out_num)
        """
        self.X = X
        return np.dot(X, self.W) + self.B
    
    def bp(self, gradient, **kwargs):
        self.B_grad = np.mean(gradient, axis=0)
        self.W_grad = np.dot(self.X.T, gradient) / gradient.shape[0]
        if not kwargs.get("backward", True):
            return
        return np.dot(gradient, self.W.T)
    
    def step(self, alpha):
        self.B -= self.B_grad * alpha
        self.W -= self.W_grad * alpha


class Conv2dLayer:
    def __init__(self, in_c, out_c, k, **kwargs):
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = kwargs.get("stride", 1)
        self.padding = kwargs.get("padding", 0)
        
        fan_in = k * k * in_c
        fan_out = k * k * out_c

        limit = np.sqrt(6 / (fan_in + fan_out))
        self.kernel = np.random.uniform(
            -limit, limit, size=(k * k * in_c, out_c)
        ).astype(np.float32)

        self.bias = np.zeros(out_c, dtype=np.float32)

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
    
    def _grad2im(self, grad):
        # grad: (N, out_c, out_h, out_w)
        N, C, H, W = self.img.shape
        k, s, oh, ow = self.k, self.stride, self.out_h, self.out_w

        grad = grad.transpose(0, 2, 3, 1).reshape(-1, self.out_c)
        kernel = self.kernel.reshape(C, k, k, self.out_c)

        dimg = np.zeros_like(self.img)

        for i in range(k):
            hs = slice(i, i + s * oh, s)
            for j in range(k):
                ws = slice(j, j + s * ow, s)

                dimg[:, :, hs, ws] += (
                    grad @ kernel[:, i, j, :].T
                ).reshape(N, oh, ow, C).transpose(0, 3, 1, 2)

        return dimg

    def __call__(self, img):
        in_c, out_c, k, stride, padding = self.in_c, self.out_c, self.k, self.stride, self.padding
        if padding > 0:
            img = np.pad(img, [(0,0), (0,0), (padding,padding), (padding,padding)], mode="constant")
        self.img = img

        self.col = self._im2col(img)                             # (batch, out_h, out_w, k*k*in_c)
        self.out_h, self.out_w = self.col.shape[1], self.col.shape[2]

        out = self.col @ self.kernel + self.bias                 # (batch, out_h, out_w, out_c)
        return out.transpose(0, 3, 1, 2)                         # (batch, out_c, out_h, out_w)

    def bp(self, gradient, **kwargs):
        # gradient: (batch, out_c, out_h, out_w)
        self.bias_grad = gradient.mean(axis=(0, 2, 3))               # (out_c,)

        # col: (batch, out_h, out_w, k*k*in_c)
        col_flat = self.col.reshape(-1, self.k * self.k * self.in_c)       # (batch*out_h*out_w, k*k*in_c)
        grad_flat = gradient.transpose(0, 2, 3, 1).reshape(-1, self.out_c) # (batch*out_h*out_w, out_c)
        self.kernel_grad = col_flat.T @ grad_flat / col_flat.shape[0]      # (k*k*in_c, out_c)

        if not kwargs.get("backward", True):
            return
        
        dimg = self._grad2im(gradient)
        if self.padding > 0:
            p = self.padding
            dimg = dimg[:, :, p:-p, p:-p]

        return dimg # (batch, in_c, in_h, in_w)

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
    A simple 2-layer neural network consisting of :
        Conv2d
        FullyConnected
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
    
    def __call__(self, x):
        """
        :params: x: input image. (batch, channel, height, weight)
        :return: predict: one-hot vector.
        """
        for calculate in self.graph:
            x = calculate(x)
        return x
    
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
