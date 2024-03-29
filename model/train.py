import pandas as pd
import torch.optim as optim
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model.model import LinearRegressionModel


class CustomCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file.
            transform (callable, optional): Optional transform to apply to the data.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Assuming the last column is the target variable
        data_row = self.data_frame.iloc[idx, :-1].values
        label = self.data_frame.iloc[idx, -1]

        # Convert to appropriate torch tensor
        data_row = torch.tensor(data_row, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            data_row = self.transform(data_row)

        return data_row, label


def train():
    dataset = CustomCSVDataset(csv_file="dataset/data.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = LinearRegressionModel()
    criterion = nn.MSELoss()

    # lower learning rate to avoid overshooting
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            y_pred = model(x_batch)

            # Compute loss
            loss = criterion(y_pred, y_batch)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()

