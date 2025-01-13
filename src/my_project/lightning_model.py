from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
import wandb

class My_model(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 32, 3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(288, 10),
            nn.Dropout(p=0.2),
            nn.Softmax(),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.backbone(x)
    
    def training_step(self, batch):
        data, target = batch
        prediction = self.forward(data)
        loss = self.criterion(prediction, target)

        preds = torch.argmax(prediction, dim=1)
        acc = (preds == target).float().mean()

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
    def test_step(self, batch):
        images, targets = batch
        outputs = self.forward(images)
        loss = self.criterion(outputs, targets)

        # Compute additional metrics if needed (e.g., accuracy)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).float().mean()

        # Log metrics
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

if __name__ == "__main__":
    model = My_model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")