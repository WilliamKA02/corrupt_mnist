from my_project.lightning_model import My_model
from pytorch_lightning import Trainer, loggers
from torch.utils.data import DataLoader
from my_project.data import corrupt_mnist
import typer
train_dataset, test_dataset = corrupt_mnist()
batch_size = 64

def train():
    model = My_model()
    trainer = Trainer(max_epochs=5, limit_train_batches=0.2, logger=loggers.WandbLogger(project="my_awesome_project"))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(model, test_dataloader)

def main():
    typer.run(train)
