import torch
import torch.nn as nn

class LanguageMLP(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=128, num_classes=5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)
    

def train_model(model, optimizer, X_train_t, y_train_t, device, epochs=30):
    criterion = nn.CrossEntropyLoss()
    losses = []
    X_train_t = X_train_t.to(device)
    y_train_t = y_train_t.to(device)

    for epoch in range(epochs):
        model.train()

        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        losses.append(loss.item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

        return losses

