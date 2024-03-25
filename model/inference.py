import torch
from train import LinearRegressionModel

model = LinearRegressionModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Predict y values
val = input("enter value: ")
x_new = torch.tensor([[float(val)]])
with torch.no_grad():
    y_pred = model(x_new)

print("prediction: ", y_pred.item())
