import torch
import numpy as np
from torch import nn

LEARNING_RATE = 0.001

class MLP(nn.Module):
    def __init__(self, input_features: int, hidden: int, output: int) -> None:
        super().__init__()
        self.lin1 = nn.Linear(input_features, hidden)
        self.lin2 = nn.Linear(hidden, output)
        
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)
        
        
    def forward(self, data: np.ndarray) -> torch.Tensor:
        x = self.lin1(data).relu()
        x = self.dropout(x)
        
        x = self.lin2(x)
        
        x = self.softmax(x)
        
        return x

    
def train(model: nn.Module, data: torch.Tensor, target: torch.Tensor) -> None:
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    model.train()

    for e in range(100):
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criterion(out, target)
        loss.backward()
        optimizer.step()
        
        pred = torch.argmax(out, dim=1)
        acc = (pred == target).sum().item() / len(target)
        
        if e % 10 == 0:
            print(f'Epoch - {e} | Loss - {loss} | Accuracy - {acc:.3f}')


def test(model: nn.Module, data: torch.Tensor, target: torch.Tensor) -> float:
    model.eval()
    out = model(data)
    
    pred = torch.argmax(out, dim=1)
    acc = (pred == target).sum().item() / len(target)
    
    return acc