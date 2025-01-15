from torch import nn, randn

model = nn.Sequential(
    nn.Conv2d(1, 64, 3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 32, 3),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(32, 32, 3),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(288, 10),
    nn.Dropout(p=0.2),
    nn.Softmax(dim=1),
)

if __name__ == "__main__":
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
