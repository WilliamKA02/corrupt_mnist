from my_project.model import model
from my_project.data import corrupt_mnist
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import typer
# import wandb
# from dotenv import load_dotenv
# import os

# Load environment variables from .env
# load_dotenv()

# Get WandB API key from the environment variable
# wandb_api_key = os.getenv("WANDB_API_KEY")
# if wandb_api_key is None:
#     raise ValueError("WANDB_API_KEY not found in environment variables. Please set it in the .env file.")

# # Log in to WandB
# wandb.login(key=wandb_api_key)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model_ = model

torch.manual_seed(42)

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    # wandb.init(project="my_awesome_project", config={"lr": lr, "batch_size": batch_size, "epochs": epochs})

    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = model_.to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()

        preds, targets = [], []
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            # wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            # print({"train_loss": loss.item(), "train_accuracy": accuracy})

            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, iter {i}, loss: {loss.item()}")

                # images = wandb.Image(img[:5].detach().cpu(), caption="Input images")
                # wandb.log({"input_images": images})
                # grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                # wandb.log({"gradients": wandb.Histogram(grads)})

    #     preds = torch.cat(preds, 0)
    #     targets = torch.cat(targets, 0)

    # final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    # final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    # final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    # final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # torch.save(model.state_dict(), "model.pth")
    # artifact = wandb.Artifact(name="corrupt_mnist_model",
    #     type="model",
    #     description="A model trained to classify corrupt MNIST images",
    #     metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    # )
    # artifact.add_file("model.pth")
    # wandb.run.log_artifact(artifact)

    print("Training complete")
    torch.save(model.state_dict(), "models/model.pth")
    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # axs[0].plot(statistics["train_loss"])
    # axs[0].set_title("Train loss")
    # axs[1].plot(statistics["train_accuracy"])
    # axs[1].set_title("Train accuracy")
    # fig.savefig("reports/figures/training_statistics.png")

# wandb.finish()

def main():
    typer.run(train)
