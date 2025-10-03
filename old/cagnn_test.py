import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    global_mean_pool,
)
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import random_split, WeightedRandomSampler
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

# # GNN Encoder for graphs
# class GNNEncoder(nn.Module):
#     def __init__(self, in_dim, hidden_dim):
#         super().__init__()
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)

#     def forward(self, x, edge_index, batch):
#         x = F.relu(self.conv1(x, edge_index))
#         x = self.conv2(x, edge_index)
#         return global_mean_pool(x, batch)  # graph-level representation

class FlexibleGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers=2,
        gnn_type='gcn',        # Options: 'gcn', 'sage', 'gat', 'gin'
        dropout=0.5,
        heads=1                # For GAT
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.gnn_type = gnn_type.lower()

        # First layer
        self.convs.append(self._build_conv(self.gnn_type, in_dim, hidden_dim, heads))
        self.bns.append(nn.BatchNorm1d(hidden_dim * (heads if self.gnn_type == 'gat' else 1)))

        # Hidden layers
        for _ in range(num_layers - 1):
            in_ch = hidden_dim * (heads if self.gnn_type == 'gat' else 1)
            out_ch = hidden_dim
            self.convs.append(self._build_conv(self.gnn_type, in_ch, out_ch, heads))
            self.bns.append(nn.BatchNorm1d(out_ch * (heads if self.gnn_type == 'gat' else 1)))

    def _build_conv(self, gnn_type, in_dim, out_dim, heads):
        if gnn_type == 'gcn':
            return GCNConv(in_dim, out_dim)
        elif gnn_type == 'sage':
            return SAGEConv(in_dim, out_dim)
        elif gnn_type == 'gat':
            return GATConv(in_dim, out_dim, heads=heads, concat=True)
        elif gnn_type == 'gin':
            return GINConv(
                nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                )
            )
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return global_mean_pool(x, batch)

# # Graph Classifier Head
# class GraphClassifier(nn.Module):
#     def __init__(self, hidden_dim, num_classes):
#         super().__init__()
#         self.fc = nn.Linear(hidden_dim, num_classes)

#     def forward(self, graph_emb):
#         return self.fc(graph_emb)

class GraphClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        # Input layer
        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, out_dim))
        else:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


# # Domain Discriminator Head
# class DomainDiscriminator(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 2)  # domain: source=0, target=1

#     def forward(self, graph_emb, alpha):
#         x = GradReverse.apply(graph_emb, alpha)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, num_layers=2, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.layers.append(nn.Linear(in_dim, 2))
        else:
            self.layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, 2))  # binary domain classification

    def forward(self, x, alpha=1.0):
        # x = GradReverse.apply(x, alpha)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

# Projection head for contrastive loss
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)

# Contrastive loss: NT-Xent (simplified)
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    batch_size = z1.size(0)

    representations = torch.cat([z1, z2], dim=0)
    similarity = torch.matmul(representations, representations.T)

    sim_ij = torch.diag(similarity, batch_size)
    sim_ji = torch.diag(similarity, -batch_size)
    positives = torch.cat([sim_ij, sim_ji], dim=0)

    nominator = torch.exp(positives / temperature)
    denominator = torch.sum(torch.exp(similarity / temperature), dim=1) - torch.exp(torch.ones_like(positives) / temperature)
    loss = -torch.log(nominator / denominator)
    return loss.mean()


# Load datasets (example: MUTAG → PROTEINS)
# src_ds = TUDataset(root='/home/mfmce/drop/data/ENZYMES', name='ENZYMES')
# tgt_ds = TUDataset(root='/home/mfmce/drop/data/PROTEINS', name='PROTEINS')

# src_loader = DataLoader(src_ds, batch_size=32, shuffle=True)
# tgt_loader = DataLoader(tgt_ds, batch_size=32, shuffle=True)

# Load full PROTEINS dataset
DATASET_NAME = 'PROTEINS'
full_ds = TUDataset(root='drop/data/'+DATASET_NAME, name=DATASET_NAME)

# Shuffle and split into two subsets
total_len = len(full_ds)
split_len = total_len // 2
src_ds, tgt_ds = random_split(full_ds, [split_len, total_len - split_len])

#----- Create weighted data loader for source data -----#

# Count class distribution
labels = torch.tensor([data.y.item() for data in src_ds])
class_counts = torch.bincount(labels)
class_weights = 1. / class_counts.float()
sample_weights = [class_weights[label] for label in labels]
# print("DEBUGGING: class_counts   = ",class_counts)
# print("DEBUGGING: class_weights  = ",class_weights)
# print("DEBUGGING: sample_weights = ",sample_weights)

sampler = WeightedRandomSampler(weights=sample_weights,
                                 num_samples=len(src_ds),
                                 replacement=True)

src_loader = DataLoader(src_ds, batch_size=32, sampler=sampler, drop_last=True)

#--------------------------------------------------------#

# Create DataLoaders
src_loader_unweighted = DataLoader(src_ds, batch_size=32, shuffle=True, drop_last=True)
tgt_loader = DataLoader(tgt_ds, batch_size=32, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
num_classes = 2
gnn_type = "gin"
heads = 4
encoder = FlexibleGNNEncoder(
    in_dim=full_ds.num_node_features,
    hidden_dim=64,
    num_layers=3,
    gnn_type=gnn_type,      # Try 'gcn', 'sage', 'gat', 'gin'
    dropout=0.4,
    heads=heads              # Only relevant for GAT
).to(device)

projector = ProjectionHead(64 * (heads if gnn_type=="gat" else 1), 32).to(device)

clf = GraphClassifier(
    in_dim=64 * (heads if gnn_type=="gat" else 1),
    out_dim=num_classes,
    num_layers=3,
    hidden_dim=128,
    dropout=0.4
).to(device)

disc = DomainDiscriminator(
    in_dim=64 * (heads if gnn_type=="gat" else 1),
    num_layers=4,
    hidden_dim=128,
    dropout=0.4
).to(device)

optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(clf.parameters()) + list(disc.parameters()),
    lr=0.001
)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Linear decay from 1.0 to 0.0 over epochs
lr_lambda = lambda epoch: (1 - (epoch / epochs))
scheduler = LambdaLR(optimizer, lr_lambda)

# # scheduler = None

def alpha_fn(epoch, total_epochs, coefficient = 0.05):
    # Default schedule: DANN-style ramp-up
    return coefficient * (2. / (1. + np.exp(-10 * epoch / total_epochs)) - 1)

alpha_fn = 0.1

# Annealed temperature
def temp_fn(epoch, max_epoch, t_min=0.07, t_max=0.5):
    return t_min + (t_max - t_min) * (1 - epoch / max_epoch)

temp_fn = 0.1

def eval_model(loader,return_labels=False):
    encoder.eval()
    clf.eval()
    correct = total = 0
    logits = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = encoder(batch.x, batch.edge_index, batch.batch)
            out = clf(emb)
            pred = F.softmax(out,dim=0).argmax(dim=1)

            # # DEEBUGGING BLOCK
            # logits = out
            # probs = F.softmax(out, dim=1)
            # print("Logits mean:", logits.mean().item())
            # print("Logits std:", logits.std().item())
            # print("Softmax max:", probs.max().item())
            # print("Softmax min:", probs.min().item())
            
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
            if return_labels:
                logits.extend(F.softmax(out,dim=0)[:,1].cpu().tolist())
                labels.extend(batch.y.cpu().tolist())
    return correct / total, logits, labels

def eval_disc(src_loader,tgt_loader,return_labels=False,alpha=1.0):
    encoder.eval()
    disc.eval()
    correct = total = 0
    logits = []
    labels = []
    with torch.no_grad():
        for src_batch, tgt_batch in zip(src_loader,tgt_loader):

            # Get source batch embedding
            src_batch = src_batch.to(device)
            src_emb = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_out = disc(src_emb)

            # Get targete batch embedding
            tgt_batch = tgt_batch.to(device)
            tgt_emb = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_out = disc(tgt_emb)

            # Domain classification loss (labels: 0 for source, 1 for target)
            domain_emb = torch.cat([src_emb, tgt_emb], dim=0)
            domain_labels = torch.cat([
                torch.zeros(src_emb.size(0), dtype=torch.long),
                torch.ones(tgt_emb.size(0), dtype=torch.long)
            ], dim=0).to(device)

            domain_pred = disc(domain_emb, alpha=alpha)
            domain_loss = F.cross_entropy(domain_pred, domain_labels)
            domain_out = F.softmax(domain_pred,dim=0).argmax(dim=1)
            correct += (domain_out == domain_labels).sum().item()
            total += domain_labels.size(0)
            if return_labels:
                logits.extend(F.softmax(domain_pred,dim=0).cpu().tolist())
                labels.extend(domain_labels.cpu().tolist())
    return correct / total, logits, domain_labels

def train(epochs=100, temp_fn=temp_fn, alpha_fn=alpha_fn):
    encoder.train()
    clf.train()

    # Set logging lists to return
    clf_losses = []
    can_losses = []
    clf_accs   = []
    lrs        = []

    # Loop training epochs
    for epoch in range(1, epochs+1):

        # Check alpha function
        if callable(alpha_fn):
            alpha = alpha_fn(epoch, epochs)
        else:
            alpha = alpha_fn

        # Check temp function
        if callable(temp_fn):
            temp = temp_fn(epoch, epochs)
        else:
            temp = temp_fn
        
        total_clf_loss = 0
        total_can_loss = 0
        # Parallel iteration over source and target loaders
        for src_batch, tgt_batch in zip(src_loader, tgt_loader):
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch = src_batch.to(device)
            src_emb = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_out = clf(src_emb)
            src_loss = F.cross_entropy(src_out, src_batch.y)

            # Target graph forward pass
            tgt_batch = tgt_batch.to(device)
            tgt_emb = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)

            # Contrastive loss (align source and target representations)
            z1 = projector(src_emb)
            z2 = projector(tgt_emb)
            can_loss = contrastive_loss(z1, z2, temperature=temp)
    
            # # Classification loss (only on source)
            # cls_loss = F.cross_entropy(src_out, src_batch.y)
    
            loss = src_loss + alpha * can_loss
            loss.backward()
            optimizer.step()

            total_clf_loss += src_loss.item()
            total_can_loss += can_loss.item()

        # Get accuracy
        src_acc, _, _ = eval_model(src_loader_unweighted)
        encoder.train()
        clf.train()

        # Append metrics for logging
        clf_losses.append(total_clf_loss)
        can_losses.append(total_can_loss)
        clf_accs.append(src_acc)

        # Log and step learning rate scheduler
        lrs.append(optimizer.param_groups[0]['lr'])
        if scheduler is not None: scheduler.step()

        print(f'Epoch {epoch:03d}  Classifier Loss: {total_clf_loss:.4f}  Contrastive Loss: {total_can_loss:.4f}')

    return clf_losses, can_losses, clf_accs, lrs

clf_losses, can_losses, src_accs, lrs = train(epochs=epochs, temp_fn=temp_fn)

src_acc, outs, ys = eval_model(src_loader_unweighted,return_labels=True)
tgt_acc, tgt_outs, _ = eval_model(tgt_loader,return_labels=True)
# dis_acc, dis_outs, _ = eval_disc(src_loader,tgt_loader,return_labels=True,alpha=1.0)
print(f'Source Accuracy: {src_acc:.4f}')
print(f'Target Accuracy: {tgt_acc:.4f}')
# print(f'Discri Accuracy: {dis_acc:.4f}')

# Temperature values
epoch_range = np.arange(1, epochs + 1)
alpha_values = [alpha_fn(e, epochs) if callable(alpha_fn) else alpha_fn for e in epoch_range]

# Plot
fig, axs = plt.subplots(2, 3, figsize=(20, 12))

# Alpha
axs[0, 0].plot(epoch_range, alpha_values, color='blue')
axs[0, 0].set_title('Contrastive Loss Coefficient',usetex=True)
axs[0, 0].set_xlabel('Epoch',usetex=True)
axs[0, 0].set_ylabel('alpha',usetex=True)

# Classifier & Domain Losses
axs[0, 1].plot(epoch_range, clf_losses, label='Classifier Loss', color='green')
axs_0_1_twinx = axs[0, 1].twinx()
axs_0_1_twinx.plot(epoch_range, can_losses, label='Constrastive Loss', color='orange')
axs[0, 1].set_title('Losses')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Classifier Loss')
axs_0_1_twinx.set_ylabel('Contrastive Loss')
axs[0, 1].legend(loc='best')
axs_0_1_twinx.legend(loc='best')

# Accuracy
axs[1, 0].plot(epoch_range, src_accs, label='Source Accuracy', color='purple')
# axs[1, 0].plot(epoch_range, tgt_acc, label='Target Accuracy', color='red')
axs[1, 0].set_title('Accuracy')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].legend()

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(ys, outs)
roc_auc = auc(fpr, tpr)

# Compute Figure of Merit: FOM = TPR / sqrt(TPR + FPR)
fom = tpr / np.sqrt(tpr + fpr + 1e-8)  # small value to avoid division by zero
best_idx = np.argmax(fom)
best_fpr, best_tpr, best_fom, best_thr = fpr[best_idx], tpr[best_idx], fom[best_idx], thresholds[best_idx]

# Plot ROC
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
axs[1, 1].scatter(best_fpr, best_tpr, color='red', marker='*', s=100, label=f'Max FOM \n(FOM={best_fom:.2f})\n(Thr={best_thr:.2f})')
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.05])
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('Classifier ROC Curve')
axs[1, 1].legend(loc="lower right")
axs[1, 1].grid(True)

# Run KS test
stat, p_value = ks_2samp(outs, tgt_outs)
print(f"KS test statistic: {stat:.4f}, p-value: {p_value:.4g}")

# Plot model outputs
axs[0, 2].hist(outs, bins=50, range=(0,1), alpha=0.6, label="Source Domain", color='skyblue', density=True)
axs[0, 2].hist(tgt_outs, bins=50, range=(0,1), alpha=0.6, label="Target Domain", color='salmon', density=True)
axs[0, 2].plot([], [], ' ', label=f"KS test statistic: {stat:.4f}, p-value: {p_value:.4g}")
axs[0, 2].set_xlim([0.0, 1.0])
axs[0, 2].set_title("Classifier Output Distribution")
axs[0, 2].set_xlabel("Predicted Probability")
axs[0, 2].set_ylabel("Density")
axs[0, 2].set_yscale('log')
axs[0, 2].legend()
axs[0, 2].grid(True)

# Learning rates
axs[1, 2].plot(epoch_range, lrs, color='blue')
axs[1, 2].set_title('Learning Rate',usetex=True)
axs[1, 2].set_xlabel('Epoch',usetex=True)
axs[1, 2].set_ylabel('Learning Rate',usetex=True)
axs[1, 2].set_yscale('log')

# # Hide unused subplot
# axs[0,2].axis('off')
# axs[1,2].axis('off')

plt.tight_layout()
plt.show()

def collect_embeddings(encoder, loader, device, domain_label):
    encoder.eval()
    all_embeds, all_labels, all_domains = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = encoder(data.x, data.edge_index, data.batch)
            all_embeds.append(x.cpu())
            all_labels.append(data.y.cpu())
            all_domains.append(torch.full((x.size(0),), domain_label))  # 0=source, 1=target

    return (
        torch.cat(all_embeds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_domains, dim=0)
    )

def plot_tsne(embeddings, labels, domains, title="t-SNE of Graph Embeddings"):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    print("DEBUGGING: here")
    embeds_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(16, 12))
    for domain in [0, 1]:  # source vs target
        for label in torch.unique(labels):
            idx = (domains == domain) & (labels == label)
            plt.scatter(
                embeds_2d[idx, 0],
                embeds_2d[idx, 1],
                label=f"{'Src' if domain==0 else 'Tgt'} - Class {label.item()}",
                alpha=0.6,
                marker = 'o' if domain==0 else '*',
                color = 'b' if label.item()==0 else 'r',
                s=20
            )

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Assuming you have:
# - encoder (trained)
# - source_loader and target_loader
# - device (cuda or cpu)

src_embeds, src_labels, src_domains = collect_embeddings(encoder, src_loader_unweighted, device, domain_label=0)
tgt_embeds, tgt_labels, tgt_domains = collect_embeddings(encoder, tgt_loader, device, domain_label=1)

# Combine
all_embeds = torch.cat([src_embeds, tgt_embeds], dim=0)
all_labels = torch.cat([src_labels, tgt_labels], dim=0)
all_domains = torch.cat([src_domains, tgt_domains], dim=0)

# Plot
plot_tsne(all_embeds.numpy(), all_labels, all_domains)
