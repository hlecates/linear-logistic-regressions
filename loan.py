
import pandas as pd

data = pd.read_csv("loan_data.csv")

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = pd.Categorical(data[column])
        data[column] = data[column].cat.codes

# Drop data with missing values
data = data.dropna()

# Normalize each column other than the "loan_status" 
for column in data.columns:
    if column != "loan_status":
        # For each column x, x = (x - x_min) / (x_max - x_min)
        data[column] = ((data[column] - data[column].min()) /
                        (data[column].max() - data[column].min()))

# Split the data into features and targets
X = data.drop("loan_status", axis=1)
Y = data['loan_status']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=42)

import torch
from torch.nn import Linear
from torch.optim import SGD

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32).type(torch.LongTensor).reshape(-1, 1)
y_test_tensor = torch.tensor(Y_test.values, dtype=torch.float32).type(torch.LongTensor).reshape(-1, 1)

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = Linear(input_dim, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Define the Negative log likelihood loss
def negative_log_likelihood(outputs, targets):
    outputs = torch.clamp(outputs, min=1e-10, max=1-1e-10)
    return -torch.sum(targets * torch.log(outputs) + (1 - targets) * torch.log(1-outputs))

model = LogisticRegressionModel(X_train_tensor.shape[1])
optimizer = SGD(model.parameters(), lr = 0.000001)

# Run gradient descent
model.train()
num_epochs = 10000

for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Forward pass
    outputs = model(X_train_tensor)
    loss = negative_log_likelihood(outputs, y_train_tensor.float())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}, training loss: {loss.item()}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).round().type(torch.LongTensor)
    accuracy = (predictions == y_test_tensor).sum() / y_test_tensor.size(0)
    print(f"Accuracy: {accuracy.item()}")