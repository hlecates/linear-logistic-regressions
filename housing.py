
# Load the dataset
import pandas as pd

data = pd.read_csv("housing.csv")

for column in data.columns:
    if data[column].dtype == 'object':
        # Convert column to categorical var
        data[column] = pd.Categorical(data[column])
        data[column] = data[column].cat.codes


# Drop data with missing values
data = data.dropna()

# Normalize the column value to be between 0 and 1
for column in data.columns:
    if column != "median_house_value":
        data[column] = ((data[column] - data[column].min()) / (data[column].max() - data[column].min()))

data["median_house_value"] = data["median_house_value"] / 1000

# Split the data into feature and target
X = data.drop("median_house_value", axis = 1)
Y = data["median_house_value"]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.01, random_state = 42)

# Train Model
import torch
from torch.nn import Linear

X_train_tensor = torch.tensor(X_train.values, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=  torch.float32)

Y_train_tensor = torch.tensor(Y_train.values, dtype = torch.float32).reshape(-1, 1)
Y_test_tensor = torch.tensor(Y_test.values, dtype = torch.float32).reshape(-1, 1)

class HousingModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(HousingModel, self).__init__()
        self.linear = Linear(input_dim, 1)


    def forward(self, x):
        return self.linear(x)

model = HousingModel(X_train.shape[1])

from torch.nn import MSELoss
from torch.optim import SGD

loss = MSELoss()
optimizer = SGD(model.parameters(), lr = 0.1)

model.train()

num_epochs = 100000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # Evaluate the training set
    outputs = model(X_train_tensor)
    # Calculate the loss
    lossV = loss(outputs, Y_train_tensor)
    # Compute the gradient 
    lossV.backward()
    optimizer.step()

    if(epoch + 1) % 100 == 0:
        print(f"epoch: {epoch + 1}, training loss: {lossV.item()}")


# Evaluate the model
model.eval()

for i, x in enumerate(X_test_tensor[:10]):
    prediction = model(x)
    print(f"Prediciton: {prediction.item()}, Actual: {Y_test_tensor[i].item()}")

# Plot the predictions vs actual values, draw a diagonal line
import matplotlib.pyplot as plt
plt.figure(figsize=(5, 5))

predictions = model(X_test_tensor)

plt.scatter(Y_test, predictions.detach().numpy())
plt.xlabel('Actual Prices (k$)')
plt.ylabel('Predicted Prices (k$)')
plt.plot([0, 600], [0, 600], color='red')
plt.show()