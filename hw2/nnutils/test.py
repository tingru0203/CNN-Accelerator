import torch
from torch.utils.data import DataLoader
import numpy as np


def test(model, dataloader: DataLoader, max_samples=None):
    cnt = 0
    total = 0
    n_inferences = 0
    for i, data in enumerate(dataloader):

        images, labels = data[0].numpy(), data[1].numpy()
        y = model.forward(images)

        y = np.argmax(y, axis=1)
        cnt = cnt + np.count_nonzero((labels == y) == True)
        total += images.shape[0]

        if max_samples:
            n_inferences += images.shape[0]
            if n_inferences >= max_samples:
                break

    print("Accuracy: {}%".format(cnt/total*100))
    return cnt/total*100
