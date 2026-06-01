import numpy as np
import matplotlib.pyplot as plt

from train import get_mnist_loaders
from model import SimpleNetwork


def plot_handwriting_predictions(images, pred_labels, labels):
    """
    images:      numpy array, shape (4, 1, 28, 28)
    pred_labels: array/list, shape (4,) or one-hot/probabilities shape (4, 10)
    labels:      array/list, shape (4,) or one-hot shape (4, 10)
    """

    # Convert one-hot/probability arrays to class labels
    if np.ndim(pred_labels) == 2:
        pred_labels = np.argmax(pred_labels, axis=1)

    if np.ndim(labels) == 2:
        labels = np.argmax(labels, axis=1)

    n = images.shape[0]

    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))

    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(images[i, 0], cmap="gray")
        ax.axis("off")

        ax.set_title(
            f"pred: {pred_labels[i]}, label: {labels[i]}"
        )

    plt.tight_layout()
    plt.savefig("output.png")


def main():
    _, test_loader = get_mnist_loaders(batch_size=4)
    model = SimpleNetwork.load("mnist_model.pkl")
    images, labels = next(iter(test_loader))

    predicts = model(images)
    pred_labels = np.argmax(predicts, axis=1)
    plot_handwriting_predictions(images, pred_labels, labels)

if __name__ == "__main__":
    main()