import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
validation_data = list(validation_data)
test_data = list(test_data)

x_train = torch.tensor(np.vstack([x.flatten() for x, _ in training_data]), dtype=torch.float32)
y_train = torch.tensor([np.argmax(y) if isinstance(y, np.ndarray) else y for _, y in training_data], dtype=torch.long)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

x_val = torch.tensor(np.vstack([x.flatten() for x, _ in training_data]), dtype=torch.float32)
y_val = torch.tensor([np.argmax(y) if isinstance(y, np.ndarray) else y for _, y in training_data], dtype=torch.long)

x_test = torch.tensor(np.vstack([x.flatten() for x, _ in training_data]), dtype=torch.float32)
y_test = torch.tensor([np.argmax(y) if isinstance(y, np.ndarray) else y for _, y in training_data], dtype=torch.long)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class FNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(FNN, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train(model, loader, optimizer, criterion, epochs, device):
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss / len(loader):.4f}")


def evaluate(model, x, y, device):
    model.eval()
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        outputs = model(x)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean().item()
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


input_size = x_train.shape[1]
output_size = 10  # MNIST has 10 classes (0-9)
layer_configs = [[128], [128, 64], [64, 64, 64]]
learning_rates = [0.1, 0.5]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = []
for hidden_sizes in layer_configs:
    for learning_rate in learning_rates:
        print(f"\nRunning experiment with layers: {hidden_sizes}, learning rate: {learning_rate}")
        model = FNN(input_size, hidden_sizes, output_size).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train(model, train_loader, optimizer, criterion, epochs=150, device=device)
        accuracy = evaluate(model, x_test, y_test, device)

        results.append({"layers": hidden_sizes, "learning_rate": learning_rate, "test_accuracy": accuracy})

for result in results:
    print(result)


