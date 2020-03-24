import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms, datasets
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from PIL import Image

class CNN(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.batch_size = batch_size
        self.lr = lr
        self.ndf = ndf

        self.val_correct_counter = 0
        self.val_total_counter = 0

        self.hidden0 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.ndf, kernel_size=4),
            nn.LeakyReLU(0.2)
        )

        self.hidden1 = nn.Sequential(
            nn.Conv2d(self.ndf, self.ndf * 4, kernel_size=4),
            nn.LeakyReLU(0.2)
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(self.ndf * 4, self.ndf, kernel_size=4),
            nn.LeakyReLU(0.2)
        )

        self.hidden3 = nn.Sequential(
            nn.Linear(5776, 1000),
            nn.LeakyReLU(0.2)
        )

        self.hidden4 = nn.Sequential(
            nn.Linear(1000, 200),
            nn.LeakyReLU(0.2)
        )

        self.hidden5 = nn.Sequential(
            nn.Linear(200, 10)
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)

        x = torch.flatten(x, start_dim=1, end_dim=-1)

        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)

        return x

    def cross_entropy_loss(self, predicted_label, label):
        return F.cross_entropy(predicted_label, label)

    def training_step(self, batch, batch_idx):
        x, y = batch

        predicted = self.forward(x)
        loss = self.cross_entropy_loss(predicted, y)

        logs = {"train_loss": loss}
        return {"loss": loss, "log": logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch

        predicted = self.forward(x)
        loss = self.cross_entropy_loss(predicted, y)

        comet_logger.experiment.log_confusion_matrix(labels=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                                                     y_true=torch.eye(10)[y].view(-1, 10),
                                                     y_predicted=predicted
                                                     )

        self.val_correct_counter += int((torch.argmax(predicted, 1).flatten() == y).sum())
        self.val_total_counter += y.size(0)

        logs = {"val_loss": loss}
        return {"val_loss": loss, "log": logs}


    def validation_epoch_end(self, outputs):
        # outputs is an array with what you returned in validation_step for each batch
        # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]

        avg_acc = 100 * self.val_correct_counter / self.val_total_counter

        self.val_correct_counter = 0
        self.val_total_counter = 0

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_acc': avg_acc, 'val_loss': avg_loss}
        return {'avg_val_acc': avg_acc, 'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self.forward(x)

        loss = self.cross_entropy_loss(y_hat, y)

        self.test_correct_counter += int((torch.argmax(y_hat, 1).flatten() == y).sum())
        self.test_total_counter += y.size(0)

        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_acc = 100 * self.test_correct_counter / self.test_total_counter

        self.test_correct_counter = 0
        self.test_total_counter = 0

        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_acc': avg_acc, 'test_loss': avg_loss}

        return {"avg_test_loss": avg_loss, "avg_test_acc": avg_acc, "log: ": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def prepare_data(self):
        compose = transforms.Compose([
            transforms.ToTensor(),
            invertColor(),
            randomNoise()
        ])

        self.mnist_train = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=compose
        )

        self.mnist_test = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=compose
        )

        self.mnist_train, self.mnist_val = torch.utils.data.random_split(self.mnist_train, [55000, 5000])

    def train_dataloader(self):
        mnist_train_loader = torch.utils.data.DataLoader(self.mnist_train,
                                                         batch_size=self.batch_size,
                                                         num_workers=1,
                                                         shuffle=True)

        return mnist_train_loader

    def val_dataloader(self):
        mnist_val_loader = torch.utils.data.DataLoader(self.mnist_val,
                                                         batch_size=self.batch_size,
                                                         num_workers=1,
                                                         shuffle=True)

        return mnist_val_loader

    def test_dataloader(self):
        mnist_test_loader = torch.utils.data.DataLoader(self.mnist_test,
                                                       batch_size=self.batch_size,
                                                       num_workers=1,
                                                       shuffle=True)

        return mnist_test_loader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

lr = 0.001
batch_size = 128*4
ndf = 16

model = CNN.load_from_checkpoint("mnist_noise_epoch=14.ckpt")
model.eval()
example = torch.rand(1, 1, 28, 28)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("app/src/main/assets/cnn_inv_noise.pt")