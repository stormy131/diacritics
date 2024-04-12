import torch
from torch.utils.data import DataLoader
import numpy as np
from torch import nn

LEARNING_RATE = 0.01

class MLP(nn.Module):
    def __init__(self, input_features: int, hidden: int, output: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_features, 2*hidden)
        self.lin2 = nn.Linear(2*hidden, hidden)
        self.lin3 = nn.Linear(hidden, output)
        
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, data: np.ndarray) -> torch.Tensor:
        x = self.lin1(data).relu()
        x = self.dropout(x)
        
        x = self.lin2(x).relu()
        x = self.dropout(x)
        
        x = self.lin3(x)
        
        x = self.softmax(x)
        
        return x

    
def train(model: nn.Module, data_loader: DataLoader) -> None:
    _, class_counts = np.unique(data_loader.dataset.target.numpy(), return_counts=True)
    class_weights = torch.tensor(
        [1 - c/len(data_loader.dataset.target) for c in class_counts]
    ).type(torch.float32)
    
    loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    model.train()
    
    for i, batch in enumerate(data_loader):
        data, target = batch
        
        optimizer.zero_grad()
        out = model(data)
        
        loss = loss_criterion(out, target)
        loss.backward()
        optimizer.step()


def test(model: nn.Module, data: torch.Tensor, target: torch.Tensor) -> float:
    model.eval()
    out = model(data.type(torch.float32))
    
    pred = torch.argmax(out, dim=1)
    acc = (pred == target).sum().item() / len(target)
    
    return acc