import os

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.nn import functional as F


class Demo(pl.LightningModule):

    def __init__(self, classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.classes)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'log': {'hits@1': 1}}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


mnist_train = MNIST(os.getcwd(), train=True, download=True,
                    transform=transforms.ToTensor())
mnist_train = DataLoader(mnist_train, batch_size=32, num_workers=4)
mnist_val = MNIST(os.getcwd(), train=True, download=True,
                  transform=transforms.ToTensor())
mnist_val = DataLoader(mnist_val, batch_size=32, num_workers=4)

model = Demo()
checkpoint_callback = ModelCheckpoint(
    filepath=os.getcwd()+'/{epoch}_{val_loss:.3f}',
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_top_k=1)

trainer = Trainer(checkpoint_callback=checkpoint_callback,
                  val_check_interval=0.1,
                  gpus='3',
                  profiler=True)

trainer.fit(model, mnist_train, mnist_val)
