import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNFeatureExtractor, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return x

class DomainClassifier(nn.Module):
    def __init__(self, in_channels):
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Binary domain classification
        )

    def forward(self, x, alpha=1.0):
        x = grad_reverse(x, alpha)
        return self.fc(x)

class JetTagger(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(JetTagger, self).__init__()
        self.feature_extractor = GNNFeatureExtractor(in_channels, hidden_channels)
        self.label_classifier = nn.Linear(hidden_channels, 1)
        self.domain_classifier = DomainClassifier(hidden_channels)

    def forward(self, x, edge_index, batch, alpha=1.0):
        features = self.feature_extractor(x, edge_index, batch)
        label_output = torch.sigmoid(self.label_classifier(features))
        domain_output = self.domain_classifier(features, alpha)
        return label_output, domain_output


import torch.optim as optim
from torch_geometric.data import DataLoader

# Initialize model, optimizer, and data loader
model = JetTagger(in_channels=2, hidden_channels=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loader = DataLoader([data], batch_size=1, shuffle=True)

# Training loop
for epoch in range(1, 21):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        label_output, domain_output = model(data.x, data.edge_index, data.batch, alpha=epoch / 20.0)
        label_loss = F.binary_cross_entropy(label_output.squeeze(), data.y.float())
        domain_loss = F.cross_entropy(domain_output, data.domain)
        loss = label_loss + domain_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch}, Loss: {total_loss / len(loader):.4f}')
