import torch
import torchvision
from torchvision import transforms
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm.notebook import tqdm
import numpy as np
import matplotlib.pyplot as plt

def load_file(path):
    return np.load(path).astype(np.float32)

train_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.49, 0.248),
                                transforms.RandomAffine(degrees=(-5, 5), translate=(0, 0.05), scale=(0.9, 1.1)),
                                transforms.RandomResizedCrop((224, 224), scale=(0.35, 1))
])
val_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.49, 0.248)
])

train_dataset = torchvision.datasets.DatasetFolder(
    "rsna-pneumonia-detection-challenge/Processed/train/",
    loader=load_file, extensions="npy", transform=train_transforms)
                                                   
val_dataset = torchvision.datasets.DatasetFolder(
    "rsna-pneumonia-detection-challenge/Processed/val/",
    loader=load_file, extensions="npy", transform=val_transforms)

batch_size = 32
num_workers = 4

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           num_workers=num_workers, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         num_workers=num_workers, persistent_workers=True, 
                                         shuffle=False)

class PneumoniaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7),
                                           stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1, bias=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # negative samples:18593,  positive samples:5407
        # 18593/5407 close to 3, so pos_weight should set to 3(tensor)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3]))
        
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        
    def forward(self, data):
        pred = self.model(data)
        return pred
    
    def training_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:,0]
        loss = self.loss_fn(pred, label)
        
        self.log("Train Loss", loss)
        self.log("Step Train ACC", self.train_acc(torch.sigmoid(pred), label.int()))
        
        return loss
      
    def on_train_epoch_end(self):
        self.log("Train ACC", self.train_acc.compute())
        
    def validation_step(self, batch, batch_idx):
        x_ray, label = batch
        label = label.float()
        pred = self(x_ray)[:,0]
        loss = self.loss_fn(pred, label)
        
        self.log("Val Loss", loss)
        self.log("Step Val ACC", self.val_acc(torch.sigmoid(pred), label.int()))
        
        return loss
    
    def on_validation_epoch_end(self):
        self.log("Val ACC", self.val_acc.compute())
        
    def configure_optimizers(self):
        return [self.optimizer]
    
model = PneumoniaModel()

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="Val ACC",
    save_top_k=10,
    mode="max")

trainer = pl.Trainer(devices=1, accelerator="gpu", logger=TensorBoardLogger(save_dir="./logs"),
                     log_every_n_steps=1, callbacks=checkpoint_callback, max_epochs=35)
                    
torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    trainer.fit(model, train_loader, val_loader)