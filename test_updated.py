#----------------------------------------------------------------------------------------------------#
# DATA
from torch_geometric.data import Data, Dataset, InMemoryDataset, download_url
import os.path as osp
from glob import glob
import multiprocessing
from tqdm import tqdm
from functools import lru_cache

# Class definitions
class SmallDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, datalist=[], clean_keys=()):
        self.datalist = datalist
        self.root = root
        self.clean_keys = clean_keys
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def clean_data(self,data):
        cleaned_data = Data()
        for key in data.keys():
            if key in (self.clean_keys): continue
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = value.detach().cpu().clone()
        return cleaned_data

    def process(self):
        
        # Check input data list
        if self.datalist is None or len(self.datalist)==0:
            return

        # Read data into huge `Data` list.
        data_list = self.datalist

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data_list = [self.clean_data(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

    # def get(self, idx):
    #     if self.datalist is None or len(self.datalist)==0: self.datalist = list(torch.load(osp.join(self.processed_dir, self.processed_file_names[0])))
    #     data = self.datalist[idx]
    #     return data

class LargeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, datalist=None, num_workers=8, chunk_size=100, pickle_protocol=5, clean_keys=('is_data', 'rec_indices')):
        self.datalist = datalist
        self.root = root
        self.num_workers = num_workers
        self.pickle_protocol = pickle_protocol
        self.chunk_size = chunk_size
        self.clean_keys = clean_keys
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.datalist is not None and len(self.datalist)>0:
            return [f'data{i}.pt' for i in range(len(self.datalist))]
        else:
            return [os.path.basename(path) for path in glob(os.path.join(self.raw_dir, '*.pt'))]

    @property
    def processed_file_names(self):
        if self.datalist is not None and len(self.datalist)>0:
            return [f'data{i}.pt' for i in range(len(self.datalist))]
        else:
            return [os.path.basename(path) for path in glob(os.path.join(self.processed_dir, '*.pt'))]

    def clean_data(self,data):

        # Create a new graph and remove undesired attributes
        cleaned_data = Data()
        for key in data.keys():
            if key in (self.clean_keys): continue
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = value.detach().cpu().clone()
        return cleaned_data

    def save_graph(self,idx):

        # Select data
        data = self.datalist[idx]

        # Apply filters and transforms
        if self.pre_filter is not None and not self.pre_filter(data):
            return
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        # Save data
        torch.save(self.clean_data(data), osp.join(self.processed_dir, self.processed_file_names[idx]), pickle_protocol=self.pickle_protocol)

    def process(self):

        # Check input data list
        if self.datalist is None or len(self.datalist)==0:
            return

        # Save graphs in several processes
        with multiprocessing.Pool(processes=min(len(self.datalist), self.num_workers)) as pool:
            try:
                list(tqdm(pool.imap_unordered(self.save_graph, range(len(self.datalist)), self.chunk_size), total=len(self.datalist)))
            except KeyboardInterrupt as e:
                print("Caught KeyboardInterrupt, terminating workers")
                pool.terminate()
                pool.join()
                print(e)
            else:
                pool.close()
                pool.join()

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

@lru_cache(maxsize=16)
def load_batch(path, weights_only=True):
    return list(torch.load(path, weights_only=weights_only))

class LazyDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, datalist=None, num_workers=8, chunk_size=100, pickle_protocol=5, clean_keys=('is_data', 'rec_indices'), batch_size=1000, use_cache=False, drop_last=False, weights_only=False):
        self.datalist = datalist
        self.root = root
        self.num_workers = num_workers
        self.pickle_protocol = pickle_protocol
        self.chunk_size = chunk_size
        self.clean_keys = clean_keys
        self.batch_size = batch_size
        self._loaded_batch = None
        self._loaded_batch_idx = -1
        self.drop_last = drop_last
        self.num_batches =  0
        if self.datalist is not None and len(self.datalist)>0 and self.batch_size>0:
            self.num_batches = len(self.datalist)//self.batch_size+(1 if len(self.datalist)%self.batch_size>0 and not self.drop_last else 0)
        self.use_cache = use_cache
        self.weights_only = weights_only
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        if self.datalist is not None and len(self.datalist)>0:
            return [f'data{i}.pt' for i in range(self.num_batches)]
        else:
            return [os.path.basename(path) for path in glob(os.path.join(self.raw_dir, 'data*.pt'))]

    @property
    def processed_file_names(self):
        if self.datalist is not None and len(self.datalist)>0:
            return [f'data{i}.pt' for i in range(self.num_batches)]
        else:
            return [os.path.basename(path) for path in glob(os.path.join(self.processed_dir, 'data*.pt'))]

    def clean_data(self,data):

        # Create a new graph and remove undesired attributes
        cleaned_data = Data()
        for key in data.keys():
            if key in (self.clean_keys): continue
            value = getattr(data, key)
            if isinstance(value, torch.Tensor):
                cleaned_data[key] = value.detach().cpu().clone()
        return cleaned_data

    def save_graph_batch(self,idx):

        # Set batch indices
        min_idx = idx*self.batch_size
        max_idx = min((idx+1)*self.batch_size, len(self.datalist)-1)

        # Select batch
        data = self.datalist[min_idx:max_idx+1]

        # Apply filters and transforms
        if self.pre_filter is not None:
            data = [d for d in data if not self.pre_filter(data)]
        if self.pre_transform is not None:
            data = [self.pre_transform(d) for d in data]

        # Clean batch
        data = [self.clean_data(d) for d in data]

        # Save batch
        torch.save(data, osp.join(self.processed_dir, self.processed_file_names[idx]), pickle_protocol=self.pickle_protocol)

    def process(self):

        # Check input data list
        if self.datalist is None or len(self.datalist)==0:
            return

        # Save batches of graphs in several processes
        with multiprocessing.Pool(processes=min(self.num_batches, self.num_workers)) as pool:
            try:
                list(tqdm(pool.imap_unordered(self.save_graph_batch, range(self.num_batches), self.chunk_size), total=self.num_batches))
            except KeyBoardInterrupt as e:
                print("Caught KeyBoardInterrupt, terminating workers")
                pool.terminate()
                pool.join()
                print(e)
            else:
                pool.close()
                pool.join()

    def len(self):
        return len(self.processed_file_names)*self.batch_size

    def get(self, idx):

        # Get indices
        batch_idx = idx // self.batch_size
        within_idx = idx % self.batch_size

        # Load batch
        if batch_idx != self._loaded_batch_idx or not self.use_cache:
            # self._loaded_batch = torch.load(os.path.join(self.processed_dir, self.processed_file_names[batch_idx]), weights_only=self.weights_only)
            # self._loaded_batch_idx = batch_idx
            # data = self._loaded_batch[within_idx]
            _loaded_batch = load_batch(os.path.join(self.processed_dir, self.processed_file_names[batch_idx]), weights_only=self.weights_only)

        # Load and transform data from batch
        data = _loaded_batch[within_idx]
        return data

def get_sampler_weights(ds):
    """
    :params:
        ds : Dataset

    :return:
        sampler weights

    :description:
        Given a labelled dataset, generate a list of weights for a sampler such that all classes are equally probable.
    """

    # Count unique labels and weight them inversely to their total counts
    labels = torch.tensor([data.y.item() for data in ds])
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts.float()
    sampler_weights = [class_weights[label] for label in labels]
    return sampler_weights

#----------------------------------------------------------------------------------------------------#
# MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    global_mean_pool,
)

# Gradient Reversal Layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

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
        x = GradReverse.apply(x, alpha)
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

#----------------------------------------------------------------------------------------------------#
# TRAIN
from tqdm import tqdm
import torch
import torch.nn.functional as F

def alpha_fn(epoch, total_epochs, coefficient = 0.05):
    return coefficient * (2. / (1. + np.exp(-10 * epoch / total_epochs)) - 1)

def temp_fn(epoch, max_epoch, t_min=0.07, t_max=0.5):
    return t_min + (t_max - t_min) * (1 - epoch / max_epoch)

def lambda_fn(epoch, epochs):
    return 2 / (1 + np.exp(-10 * epoch / epochs)) - 1

def contrastive_loss(z1, z2, temperature=0.5):

    # Compute the contrastive loss: NT-Xent (simplified)
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

#----- TIToK Loss definitions -----#

def exploss(y_source_prob, y_source, alpha=0.5):

    # Compute the exponential loss
    loss_sum = 0
    nc = y_source_prob.size(1)
    for i in range(nc):
        index_i = y_source == i
        a = torch.exp(-alpha * y_source_prob[index_i, i])
        b = 0
        for j in range(nc):
            if j == i:
                continue
            index_j = y_source == j
            ni = index_i.float().sum().item()
            nj = index_j.float().sum().item()
            if ni > 0 and nj > 0:
                b += torch.sum(torch.exp(alpha * y_source_prob[index_j, i])) / (ni * nj)
        loss_sum += torch.sum(a) * b

    return loss_sum

def soft_label_loss(tgt_logits, soft_labels_batch, temperature=2.0):

    # Compute the soft label loss
    loss_soft = torch.zeros(())
    output = F.softmax(tgt_logits / temperature, dim=1)
    if float(output.size(0)) > 0:
        loss_soft = -torch.sum(soft_labels_batch * torch.log(output)) / output.size(0)

    return loss_soft

def gaussian_kernel(x, y, kernel_mul=2.0, kernel_num=5):
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), -1, -1)
    total1 = total.unsqueeze(1).expand(-1, total.size(0), -1)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    bandwidth = torch.sum(L2_distance.data) / (total.size(0) ** 2 - total.size(0))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]

    return sum(kernel_val)

def mmd_loss(source, target):

    # Compute the Maximum Mean Discrepancy loss
    batch_size = source.size(0)
    kernels = gaussian_kernel(source, target)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)

    return loss

def gen_soft_labels(num_classes, loader, encoder, clf, temperature=2, device='cuda:0'): #NOTE: FROM: https://github.com/big-chuan-bro/TiTok/blob/main/ImbalancedDA/Titok.py
    
    # Set models to evaluation mode
    encoder.eval()
    clf.eval()

    # Create arrays
    soft_labels = torch.zeros(num_classes, 1, num_classes).to(device)
    sum_classes = torch.zeros(num_classes).to(device)
    preds_total = []
    label_total = []

    # Loop data
    for batch in loader:

        # Apply model
        batch  = batch.to(device)
        feats  = encoder(batch.x, batch.edge_index, batch.batch)
        logits = clf(feats)

        # Add to arrays
        label_total.append(batch.y)
        preds = F.softmax(logits / temperature, dim=1).data.to(device)
        preds_total.append(preds)

    # Concatenate arrays
    preds_total = torch.cat(preds_total)
    label_total = torch.cat(label_total)

    # Loop data and set class counts and soft labels
    for i in range(len(loader)):
        sum_classes[label_total[i]] += 1
        soft_labels[label_total[i]][0] += preds_total[i]

    # Loop classes and divide soft labels by class counts
    for cl_idx in range(num_classes):
        soft_labels[cl_idx][0] /= sum_classes[cl_idx]

    return soft_labels

def ret_soft_label(label, soft_labels, num_classes=2, device='cuda:0'): #NOTE: FROM: https://github.com/big-chuan-bro/TiTok/blob/main/ImbalancedDA/Titok.py
    
    # Compute the soft label for a batch
    soft_label_for_batch = torch.zeros(label.size(0), num_classes).to(device)
    for i in range(label.size(0)):
        soft_label_for_batch[i] = soft_labels[label.data[i]]

    return soft_label_for_batch

def loss_titok(src_feats, src_logits, src_labels, tgt_feats, tgt_logits, soft_labels, loss_auc_alpha=0.5, loss_soft_temperature=2.0, confidence_threshold=0.8, num_classes=2, pretraining=False, device='cuda:0', coeff_mmd=0.3, lambd=1.0, coeff_auc=0.01, coeff_soft=0.25):

    # Source classification loss
    loss_cls = F.cross_entropy(src_logits, src_labels)

    # Check if pretraining for soft labels
    if pretraining:
        return loss_cls, loss_cls, torch.zeros(()), torch.zeros(()), torch.zeros(())

    # Apply softmax to get probabilities on target domain
    tgt_probs = F.softmax(tgt_logits, dim=1)

    # Get max class probabilities
    confidences, pred_classes = torch.max(tgt_probs, dim=1)  # [B]

    # Select samples with confidence above a threshold
    mask = confidences >= confidence_threshold

    # Select the logits and predicted labels of those confident samples
    tgt_logits_confident = tgt_logits[mask]           # [B_confident, num_classes]
    tgt_labels_confident = pred_classes[mask]         # [B_confident]

    # Get soft labels
    soft_labels_batch = ret_soft_label(tgt_labels_confident, soft_labels, num_classes=num_classes, device=device)
    
    # AUC-style loss (exploss)
    loss_auc = exploss(F.softmax(src_logits, dim=1), src_labels, alpha=loss_auc_alpha)
    
    # Optional: MMD loss between source/target embeddings
    loss_mmd = mmd_loss(src_feats, tgt_feats)
    
    # Target knowledge distillation loss (on confident samples only)
    loss_soft = soft_label_loss(tgt_logits_confident, soft_labels_batch, temperature=loss_soft_temperature)
    
    # Combine losses
    loss = loss_cls + coeff_mmd * lambd * loss_mmd + coeff_auc * loss_auc + coeff_soft * loss_soft

    return loss, loss_cls, loss_mmd, loss_auc, loss_soft

def train(epochs=100, alpha_fn=0.1):
    encoder.train()
    clf.train()
    disc.train()

    # Set logging lists to return
    clf_losses = []
    dom_losses = []
    clf_accs   = []
    lrs        = []

    # Loop training epochs
    for epoch in range(1, epochs+1):

        # Check alpha function
        if callable(alpha_fn):
            alpha = alpha_fn(epoch, epochs)
        else:
            alpha = alpha_fn
        
        total_clf_loss = 0
        total_domain_loss = 0
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

            # Domain classification loss (labels: 0 for source, 1 for target)
            domain_emb = torch.cat([src_emb, tgt_emb], dim=0)
            domain_labels = torch.cat([
                torch.zeros(src_emb.size(0), dtype=torch.long),
                torch.ones(tgt_emb.size(0), dtype=torch.long)
            ], dim=0).to(device)

            domain_pred = disc(domain_emb, alpha)
            domain_loss = F.cross_entropy(domain_pred, domain_labels)

            loss = src_loss + domain_loss
            loss.backward()
            optimizer.step()

            total_clf_loss += src_loss.item()
            total_domain_loss += domain_loss.item()

        # Get accuracy
        src_acc, _, _ = eval_model(src_loader)
        encoder.train()
        clf.train()

        # Append metrics for logging
        clf_losses.append(total_clf_loss)
        dom_losses.append(total_domain_loss)
        clf_accs.append(src_acc)

        # Log and step learning rate scheduler
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

        print(f'Epoch {epoch:03d}  Classifier Loss: {total_clf_loss:.4f}  Discriminator Loss: {total_domain_loss:.4f}')

    return clf_losses, dom_losses, clf_accs, lrs

def train_can(epochs=100, temp_fn=temp_fn, alpha_fn=0.1):
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

def train_titok(encoder, clf, src_train_loader, tgt_train_loader, src_val_loader, tgt_val_loader, num_classes=2, soft_labels_temp=2, nepochs=100, confidence_threshold=0.8, temp_fn=0.1, alpha_fn=0.1, lambda_fn=lambda_fn, coeff_mmd=0.3, coeff_auc=0.01, coeff_soft=0.25, pretrain_frac=0.2, device='cuda:0', verbose=True):

    # Create soft labels #NOTE: Pretrain first
    soft_labels = None
    if pretrain_frac<=0.0:
        soft_labels = gen_soft_labels(
            num_classes, src_train_loader, encoder, clf, temperature=soft_labels_temp, device=device
        )

    # Set models in train mode
    encoder.train()
    clf.train()

    # Set logging lists to return
    logs = {}
    logs['train_losses']         = []
    logs['train_losses_cls']     = []
    logs['train_losses_auc']     = []
    logs['train_losses_mmd']     = []
    logs['train_losses_soft']    = []
    logs['train_accs_raw']       = []
    logs['train_accs_per_class'] = []
    logs['train_accs_balanced']  = []
    logs['val_losses']           = []
    logs['val_losses_cls']       = []
    logs['val_losses_auc']       = []
    logs['val_losses_mmd']       = []
    logs['val_losses_soft']      = []
    logs['val_accs_raw']         = []
    logs['val_accs_per_class']   = []
    logs['val_accs_balanced']    = []
    logs['lrs']                  = []

    # Loop training epochs
    for epoch in tqdm(range(1, nepochs+1)):

        # Check alpha function
        if callable(alpha_fn):
            alpha = alpha_fn(epoch, nepochs)
        else:
            alpha = alpha_fn

        # Check temp function
        if callable(temp_fn):
            temp = temp_fn(epoch, nepochs)
        else:
            temp = temp_fn

        # Check lambda function
        if callable(lambda_fn):
            lambd = lambda_fn(epoch, nepochs)
        else:
            lambd = lambda_fn

        # Set soft labels after pretraining
        pretraining = (epoch/nepochs<=pretrain_frac and pretrain_frac>0.0)
        if soft_labels is None and not pretraining:
            soft_labels = gen_soft_labels(num_classes, src_train_loader, encoder, clf, temperature=soft_labels_temp, device=device)

        # Iterate over source and target loaders in parallel
        for src_batch, tgt_batch in zip(src_train_loader, tgt_train_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch  = src_batch.to(device)
            src_feats  = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_labels = src_batch.y

            # Target graph forward pass
            tgt_batch  = tgt_batch.to(device)
            tgt_feats  = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_logits = clf(tgt_feats)

            # Compute loss
            loss, loss_cls, loss_mmd, loss_auc, loss_soft = loss_titok(src_feats, src_logits, src_labels, tgt_feats, tgt_logits, soft_labels, loss_auc_alpha=0.5, loss_soft_temperature=2.0, confidence_threshold=confidence_threshold, pretraining=pretraining, num_classes=num_classes, device=device)

            # Backpropagate losses and update parameters
            loss.backward()
            optimizer.step()

        # Evaluate on training and vallidation data and then put model back in training mode
        train_logs = val_titok(
            encoder, clf, src_train_loader, tgt_train_loader, soft_labels,
            pretraining=pretraining, num_classes=num_classes, confidence_threshold=confidence_threshold,
            temp=temp, alpha=alpha, lambd=lambd, coeff_mmd=coeff_mmd, coeff_auc=coeff_auc,
            coeff_soft=coeff_soft, device=device, verbose=verbose
        )
        val_logs = val_titok(
            encoder, clf, src_val_loader, tgt_val_loader, soft_labels,
            pretraining=pretraining, num_classes=num_classes, confidence_threshold=confidence_threshold,
            temp=temp, alpha=alpha, lambd=lambd, coeff_mmd=coeff_mmd, coeff_auc=coeff_auc,
            coeff_soft=coeff_soft, device=device, verbose=verbose
        )
        encoder.train()
        clf.train()

        # Append metrics for logging
        logs['train_losses'].append(train_logs["loss"])
        logs['train_losses_cls'].append(train_logs["loss_cls"])
        logs['train_losses_mmd'].append(train_logs["loss_mmd"])
        logs['train_losses_auc'].append(train_logs["loss_auc"])
        logs['train_losses_soft'].append(train_logs["loss_soft"])
        logs['train_accs_raw'].append(train_logs["acc_raw"])
        logs['train_accs_per_class'].append(train_logs["acc_per_class"])
        logs['train_accs_balanced'].append(train_logs["acc_balanced"])
        logs['val_losses'].append(val_logs["loss"])
        logs['val_losses_cls'].append(val_logs["loss_cls"])
        logs['val_losses_mmd'].append(val_logs["loss_mmd"])
        logs['val_losses_auc'].append(val_logs["loss_auc"])
        logs['val_losses_soft'].append(val_logs["loss_soft"])
        logs['val_accs_raw'].append(val_logs["acc_raw"])
        logs['val_accs_per_class'].append(val_logs["acc_per_class"])
        logs['val_accs_balanced'].append(val_logs["acc_balanced"])
        logs['lrs'].append(optimizer.param_groups[0]['lr'])

        # Step learning rate step scheduler
        if scheduler is not None: scheduler.step()

        # Print training info
        if verbose:
            message = [f'Epoch {epoch:03d}']
            for key in logs:                
                if type(logs[key][-1])==float:
                    message.append(f'{key}: {logs[key][-1]:.4f}')
            message = '\n\t'.join(message)
            print(message)

    return logs, soft_labels
    
#----------------------------------------------------------------------------------------------------#
# EVAL
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc

def val_titok(encoder, clf, src_val_loader, tgt_val_loader, soft_labels, pretraining=False, return_labels=True, num_classes=2, confidence_threshold=0.8, temp=1.0, alpha=1.0, lambd=1.0, coeff_mmd=0.3, coeff_auc=0.01, coeff_soft=0.25, device='cuda:0', verbose=True):
    
    # Set models in eval mode
    encoder.eval()
    clf.eval()
        
    # Initialize variables
    total_loss        = 0
    total_loss_cls    = 0
    total_loss_auc    = 0
    total_loss_mmd    = 0
    total_loss_soft   = 0
    correct           = 0
    total             = 0
    all_src_probs     = []
    all_src_preds     = []
    all_src_labels    = []
    correct_per_class = torch.zeros(num_classes).to(device)
    total_per_class   = torch.zeros(num_classes).to(device)

    # Iterate over source and target loaders in parallel
    with torch.no_grad():
        for src_batch, tgt_batch in zip(src_val_loader, tgt_val_loader):

            # Reset gradients
            optimizer.zero_grad()

            # Source graph forward pass
            src_batch  = src_batch.to(device)
            src_feats  = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = clf(src_feats)
            src_probs  = F.softmax(src_logits,dim=1)
            src_preds  = src_probs.argmax(dim=1)
            src_labels = src_batch.y

            # Target graph forward pass
            tgt_batch  = tgt_batch.to(device)
            tgt_feats  = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_logits = clf(tgt_feats)

            # Compute loss
            loss, loss_cls, loss_mmd, loss_auc, loss_soft = loss_titok(
                src_feats, src_logits, src_labels, tgt_feats, tgt_logits, soft_labels,
                loss_auc_alpha=0.5, loss_soft_temperature=2.0, confidence_threshold=confidence_threshold,
                num_classes=num_classes, pretraining=pretraining, device=device
            )

            # Pop losses
            total_loss      += loss.item()
            total_loss_cls  += loss_cls.item()
            total_loss_mmd  += loss_mmd.item()
            total_loss_auc  += loss_auc.item()
            total_loss_soft += loss_soft.item()

            # Count correct predictions
            correct += (src_preds == src_labels).sum().item()
            total   += src_labels.size(0)
            if return_labels:
                all_src_probs.extend(src_probs.cpu().tolist())
                all_src_preds.extend(src_preds.cpu().tolist())
                all_src_labels.extend(src_labels.cpu().tolist())

            for i in range(len(src_preds)):
                label = src_labels[i]
                total_per_class[label] += 1
                if src_preds[i] == label:
                    correct_per_class[label] += 1

    # Compute per-class accuracies, avoiding division by zero
    acc_per_class = correct_per_class / (total_per_class + 1e-8)

    # Compute average per-class accuracy
    valid_class_mask = total_per_class > 0
    acc_balanced = acc_per_class[valid_class_mask].mean().item()

    # Compute raw accuracy
    acc_raw = correct / total

    # Convert lists to torch tensors
    all_src_probs = torch.tensor(all_src_probs)
    all_src_preds = torch.tensor(all_src_preds)
    all_src_labels = torch.tensor(all_src_labels)

    logs = {
        "loss": total_loss,
        "loss_cls": total_loss_cls,
        "loss_mmd": total_loss_mmd,
        "loss_auc": total_loss_auc,
        "loss_soft": total_loss_soft,
        "acc_raw": acc_raw,
        "acc_per_class": acc_per_class.cpu().tolist(),
        "acc_balanced": acc_balanced,
        "probs": all_src_probs,
        "preds": all_src_preds,
        "labels": all_src_labels,
    }

    return logs

def eval_disc(src_loader,tgt_loader,return_labels=False):

    # Set models to evaluation mode
    encoder.eval()
    disc.eval()

    # Initialize variables and arrays
    loss    = 0
    correct = 0
    total   = 0
    probs   = []
    preds   = []
    labels  = []

    # Loop source and target domain data
    with torch.no_grad():
        for src_batch, tgt_batch in zip(src_loader,tgt_loader):

            # Get source batch embedding and logits
            src_batch  = src_batch.to(device)
            src_feats  = encoder(src_batch.x, src_batch.edge_index, src_batch.batch)
            src_logits = disc(src_feats)

            # Get target batch embedding and logits
            tgt_batch  = tgt_batch.to(device)
            tgt_feats  = encoder(tgt_batch.x, tgt_batch.edge_index, tgt_batch.batch)
            tgt_logits = disc(tgt_feats)

            # Get domain classification predictions and loss
            dom_feats  = torch.cat([src_feats, tgt_feats], dim=0)
            dom_labels = torch.cat([
                torch.zeros(src_feats.size(0), dtype=torch.long),
                torch.ones(tgt_feats.size(0), dtype=torch.long)
            ], dim=0).to(device)
            dom_logits = disc(dom_feats, alpha=alpha)
            dom_loss   = F.cross_entropy(dom_logits, dom_labels)
            dom_probs  = F.softmax(dom_logits,dim=0)
            dom_preds  = dom_probs.argmax(dim=1)

            # Record total loss
            loss += dom_loss.item()

            # Record domain correct predictions, logits, and labels
            correct += (dom_preds == dom_labels).sum().item()
            total   += dom_labels.size(0)
            if return_labels:
                probs.extend(dom_probs.cpu().tolist())
                preds.extend(dom_preds.cpu().tolist())
                labels.extend(dom_labels.cpu().tolist())

        # Compute accuracy
        acc = correct / total

        # Convert lists to torch tensors
        probs = torch.tensor(probs)
        preds = torch.tensor(preds)
        labels = torch.tensor(labels)

    logs = {
        'loss':loss,
        'acc':acc,
        'probs':probs,
        'preds':preds,
        'dom_labels':dom_labels
    }

    return logs

def get_best_threshold(labels, probs):

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # Compute Figure of Merit: FOM = TPR / sqrt(TPR + FPR)
    fom = tpr / np.sqrt(tpr + fpr + 1e-8)  # small value to avoid division by zero
    best_idx = np.argmax(fom)
    best_fpr, best_tpr, best_fom, best_thr = fpr[best_idx], tpr[best_idx], fom[best_idx], thresholds[best_idx]

    logs = {
        'fpr':fpr,
        'tpr':tpr,
        'roc_auc':roc_auc,
        'best_fpr':best_fpr,
        'best_tpr':best_tpr,
        'best_fom':best_fom,
        'best_thr':best_thr
    }

    return logs, thresholds

#----------------------------------------------------------------------------------------------------#
# PLOT
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE

# Plot metrics by epoch
def plot_epoch_metrics(ax, nepochs, title='', xlabel='', ylabel='', yscale=None, xscale=None, legend_bbox_to_anchor=(1.05, 1), legend_loc='upper left', epoch_metrics=[], plot_kwargs=[], normalize_to_max=True):
    
    # Check dimensions of metrics and plotting arguments lists
    if len(epoch_metrics)!=len(plot_kwargs):
        raise ValueError(f"Number of epoch metrics ({len(epoch_metrics)}) does not match number of plot kwargs ({len(plot_kwargs)})")

    # Loop and plot metrics
    for idx, epoch_metric in enumerate(epoch_metrics):
        ax.plot(range(nepochs), epoch_metric/np.max(epoch_metric) if normalize_to_max else epoch_metric, **plot_kwargs[idx])

    # Set up plot
    ax.set_title(title, usetex=True)
    ax.set_xlabel(xlabel, usetex=True)
    ax.set_ylabel(ylabel, usetex=True)
    if yscale is not None:
        ax.set_yscale(yscale)
    if xscale is not None:
        ax.set_xscale(xscale)
    if legend_loc is not None and legend_bbox_to_anchor is None:
        ax.legend(loc=legend_loc)
    if legend_loc is not None and legend_bbox_to_anchor is not None:
        if np.any([el>1.0 or el<0.0 for el in legend_bbox_to_anchor]):
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)

# Plot ROC
def plot_roc(ax, fpr=[], tpr=[], roc_auc=0.0, best_fpr=0.0, best_tpr=0.0, best_fom=0.0, best_thr=0.0):
    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.scatter(best_fpr, best_tpr, color='red', marker='*', s=100, label=f'Max FOM \n(FOM={best_fom:.2f})\n(Thr={best_thr:.2f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Classifier ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(True)

# Plot domain predictions with KS statistic
def plot_domain_preds(ax, src_preds, tgt_preds, bins=50):
    stat, p_value = ks_2samp(src_preds, tgt_preds)
    ax.hist(src_preds, bins=bins, range=(0, 1), alpha=0.6, label="Source Domain", color='skyblue', density=True)
    ax.hist(tgt_preds, bins=bins, range=(0, 1), alpha=0.6, label="Target Domain", color='salmon', density=True)
    ax.plot([], [], ' ', label=f"KS test statistic: {stat:.4f}, p-value: {p_value:.4g}")
    ax.set_xlim([0.0, 1.0])
    ax.set_title("Classifier Output Distribution")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)

def collect_embeddings(encoder, clf, loader, device, domain_label):
    encoder.eval()
    clf.eval()
    all_embeds, all_labels, all_domains, all_preds = [], [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            x = encoder(data.x, data.edge_index, data.batch)
            logits = clf(x)
            preds = F.softmax(logits,dim=0).argmax(dim=1)
            all_embeds.append(x.cpu())
            all_labels.append(data.y.cpu())
            all_domains.append(torch.full((x.size(0),), domain_label))  # 0=source, 1=target
            all_preds.append(preds.cpu())

    return (
        torch.cat(all_embeds, dim=0),
        torch.cat(all_labels, dim=0),
        torch.cat(all_domains, dim=0),
        torch.cat(all_preds, dim=0)
    )

def plot_tsne(ax, embeddings, labels, domains, title="t-SNE of Graph Embeddings"):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)

    embeds_2d = tsne.fit_transform(embeddings)

    for domain in [0, 1]:  # source vs target
        for label in torch.unique(labels):
            idx = (domains == domain) & (labels == label)
            ax.scatter(
                embeds_2d[idx, 0],
                embeds_2d[idx, 1],
                label=f"{'Src' if domain==0 else 'Tgt'} - Class {label.item()}",
                alpha=0.6,
                marker = 'o' if domain==0 else '*',
                color = 'b' if label.item()==0 else 'r',
                s=20
            )

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(title)

def get_kinematics(encoder, clf, dataloader, threshold=0.7, device='cuda',
                                  class_idx_signal=1, class_idx_background=0):
    """
    Plots histograms of each kinematic variable for predicted signal and background.
    """
    encoder.eval()
    clf.eval()
    all_sg_kin = []
    all_bg_kin = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            feats = encoder(data.x, data.edge_index, data.batch)
            logits = clf(feats)
            probs = F.softmax(logits, dim=1)
            pred_probs, pred_classes = probs.max(dim=1)

            # Apply threshold selection
            selected = pred_probs >= threshold
            selected_classes = pred_classes[selected]
            selected_kinematics = data.kinematics[selected]

            for k, cls in zip(selected_kinematics, selected_classes):
                if cls.item() == class_idx_signal:
                    all_sg_kin.append(k.cpu())
                elif cls.item() == class_idx_background:
                    all_bg_kin.append(k.cpu())

    if not all_sg_kin or not all_bg_kin:
        print("Not enough events passed the threshold to plot.")
        return all_sg_kin, all_bg_kin

    # Convert to tensors
    sg_kin = torch.stack(all_sg_kin)  # [n_sg, n_kin]
    bg_kin = torch.stack(all_bg_kin)        # [n_bg, n_kin]

    return sg_kin, bg_kin

def plot_kinematics(axs, sg_kin, bg_kin, kin_indices=None, kin_xlabels=None,
                    sg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Signal', 'color':'C0', 'density':True},
                    bg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Background', 'color':'C1', 'density':True}):
    
    # Set number of kinematics
    n_kin = sg_kin.size(1) if type(sg_kin)==torch.Tensor else 0
    if kin_indices is None:
        kin_indices = [i for i in range(n_kin)]

    if n_kin<len(kin_indices) or len(kin_indices)!=len(kin_xlabels):
        raise ValueError(
            'Number of kinematics is not consistent ' +
            f'sg_kin.size(1) = {n_kin:d} ,' +
            f'len(kin_indices) = {len(kin_indices):d} ,' +
            f'len(kin_xlabels) = {len(kin_xlabels):d}')

    # Set kinematics labels
    if kin_xlabels is None:
        kin_xlabels = [f"Kin_{i}" for i in kin_indices]

    # Set and flatten axes
    if axs is None or len(axs)==0:
        fig, axs = plt.subplots(nrows=(len(kin_indices) + 1) // 2, ncols=2, figsize=(14, 4 * ((len(kin_indices) + 1) // 2)))
    axs = axs.flatten()

    # Turn off unused axes
    for idx in range(len(axs) - len(kin_indices)):
        axs[-1 - idx].axis('off')

    # Loop and plot kinematics
    for i, kin_idx in enumerate(kin_indices):
        axs[i].hist(sg_kin[:, kin_idx], **sg_hist_kwargs)
        axs[i].hist(bg_kin[:, kin_idx], **bg_hist_kwargs)
        axs[i].set_xlabel(kin_xlabels[i],usetex=True)
        axs[i].legend()

    return fig, axs

#----------------------------------------------------------------------------------------------------#
# UI
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import random_split, WeightedRandomSampler
import os.path as osp
from os import makedirs
import json

def pipeline_titok(
    is_tudataset = False,
    out_dir = '',
    dataset_name = None, #Note attempt to load TUDataset if given
    transform = None, #T.Compose([T.ToUndirected(),T.KNNGraph(k=6),T.NormalizeFeatures()]),
    max_idx = 1000,
    src_root='src_dataset/',
    tgt_root='tgt_dataset/',

    # loader arguments
    batch_size = 32,
    drop_last = True,

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    nepochs = 200,
    num_classes = 2,
    gnn_type = "gin",
    hdim_gnn = 64,
    num_layers_gnn = 3,
    dropout_gnn = 0.4,
    heads = 4,
    num_layers_clf = 3,
    hdim_clf = 128,
    dropout_clf = 0.4,

    # Learning rate arguments
    lr=0.001,
    lr_scheduler = 'linear', # None/'', step, and linear
    lr_kwargs = {'step_size':10, 'gamma':0.5}, #NOTE: default for step

    soft_labels_temp=2,
    confidence_threshold=0.8,
    temp_fn=1.0,
    alpha_fn=1.0,
    lambda_fn=lambda_fn,
    coeff_mmd=0.3,
    coeff_auc=0.01,
    coeff_soft=0.25,
    pretrain_frac=0.2,
    verbose=False,

    return_labels=True,
    pretraining=False,
    metrics_plot_path = 'metrics_plot.pdf',
    metrics_plot_figsize=(24,12),
    logs_path = 'logs.json',
    tsne_plot_path = 'tsne_plot.pdf',
    tsne_plot_figsize=(20,8),

    # Plot kinematics arguments
    kin_indices = [i for i in range(3,11)],
    kin_xlabels = ['$Q^2$ (GeV$^2$)', '$\\nu$', '$W$ (GeV)', '$x$', '$y$', '$z_{p\\pi^{-}}$', '$x_{F p\\pi^{-}}$', '$M_{p\\pi^{-}}$ (GeV)'], # 'idxe', 'idxp', 'idxpi', 
    best_thr = roc_info['best_thr'],
    src_kinematics_plot_path = 'src_kinematics_plot.pdf',
    tgt_kinematics_plot_path = 'tgt_kinematics_plot.pdf',
    kinematics_axs = None
    ):

    # Create output directory
    if out_dir is not None and len(out_dir)>0:
        makedirs(out_dir, exist_ok=True)

    # Load TUDataset or custom dataset
    src_ds, tgt_ds = None, None
    if is_tudataset:
        
        # Shuffle and split into two subsets
        if src_root==tgt_root or tgt_root is None or len(tgt_root)==0:
            src_root_exp = osp.expanduser(src_root)
            full_ds = TUDataset(root=osp.dirname(src_root_exp), name=osp.basename(src_root_exp))
            total_len = len(full_ds)
            split_len = total_len // 2
            src_ds, tgt_ds = random_split(full_ds, [split_len, total_len - split_len])

        # Or load two datasets
        else:
            src_root_exp = osp.expanduser(src_root)
            tgt_root_exp = osp.expanduser(tgt_root)
            src_ds = TUDataset(root=osp.dirname(src_root_exp), name=osp.basename(src_root_exp))
            tgt_ds = TUDataset(root=osp.dirname(tgt_root_exp), name=osp.basename(tgt_root_exp))

    # Load a custom pyg dataset
    else:

        #----- Load datasets -----#
        src_ds = SmallDataset(
                src_root,
                transform=transform, 
                pre_transform=None,
                pre_filter=None
            )[0:max_idx]
        
        tgt_ds = SmallDataset(
                tgt_root,
                transform=transform,
                pre_transform=None,
                pre_filter=None
            )[0:max_idx]

    #----- Create weighted data loader for source and target data -----#

    sampler_train_weights = get_sampler_weights(src_train_ds)

    sampler_train = WeightedRandomSampler(weights=sampler_train_weights,
                                    num_samples=len(src_train_ds),
                                    replacement=True)

    # Create DataLoaders
    src_train_loader = DataLoader(src_train_ds, batch_size=batch_size, sampler=sampler_train, drop_last=drop_last)
    src_train_loader_unweighted = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    tgt_train_loader = DataLoader(tgt_train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    #----- Create weighted data loader for source and target data -----#

    sampler_val_weights = get_sampler_weights(src_val_ds)

    sampler_val = WeightedRandomSampler(weights=sampler_val_weights,
                                    num_samples=len(src_val_ds),
                                    replacement=True)

    # Create DataLoaders
    src_val_loader = DataLoader(src_val_ds, batch_size=batch_size, sampler=sampler_val, drop_last=drop_last)
    src_val_loader_unweighted = DataLoader(src_val_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    tgt_val_loader = DataLoader(tgt_val_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    #--------------------------------------------------------#
    # Create model

    num_node_features = src_ds[0].num_node_features

    encoder = FlexibleGNNEncoder(
        in_dim=num_node_features,
        hidden_dim=hdim_gnn,
        num_layers=num_layers_gnn,
        gnn_type=gnn_type,      # Try 'gcn', 'sage', 'gat', 'gin'
        dropout=dropout_gnn,
        heads=heads              # Only relevant for GAT
    ).to(device)

    clf = GraphClassifier(
        in_dim=hdim_gnn * (heads if gnn_type=="gat" else 1),
        out_dim=num_classes,
        num_layers=num_layers_clf,
        hidden_dim=hdim_clf,
        dropout=dropout_clf
    ).to(device)

    #---------- Set optimizer and learning rate scheduler ----------#
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(clf.parameters()),
        lr=lr
    )
    scheduler = None
    if lr_scheduler=='step':
        scheduler = StepLR(optimizer, **lr_args)
    if lr_scheduler=='linear':
        lr_lambda = lambda epoch: (1 - (epoch / nepochs))
        scheduler = LambdaLR(optimizer, lr_lambda)

    #----- Train model
    train_logs, soft_labels = train_titok(
        encoder,
        clf,
        src_train_loader,
        tgt_train_loader,
        src_val_loader,
        tgt_val_loader,
        num_classes=num_classes,
        soft_labels_temp=soft_labels_temp,
        nepochs=nepochs,
        confidence_threshold=confidence_threshold,
        temp_fn=temp_fn,
        alpha_fn=alpha_fn,
        lambda_fn=lambda_fn,
        coeff_mmd=coeff_mmd,
        coeff_auc=coeff_auc,
        coeff_soft=coeff_soft,
        pretrain_frac=pretrain_frac,
        device=device,
        verbose=verbose
    )

    #----- Test model
    temp = temp_fn if not callable(temp_fn) else temp_fn(nepochs,nepochs)
    alpha = alpha_fn if not callable(alpha_fn) else alpha_fn(nepochs,nepochs)
    lambd = lambda_fn if not callable(lambda_fn) else lambda_fn(nepochs,nepochs)
    src_val_logs = val_titok(
        encoder,
        clf,
        src_val_loader,
        tgt_val_loader,
        soft_labels,
        return_labels=return_labels,
        pretraining=pretraining,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        temp=temp,
        alpha=alpha,
        lambd=lambd,
        coeff_mmd=coeff_mmd,
        coeff_auc=coeff_auc,
        coeff_soft=coeff_soft,
        device=device,
        verbose=verbose
    )

    tgt_val_logs = val_titok(
        encoder,
        clf,
        tgt_val_loader,
        tgt_val_loader,
        soft_labels,
        return_labels=return_labels,
        pretraining=pretraining,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
        temp=temp,
        alpha=alpha,
        lambd=lambd,
        coeff_mmd=coeff_mmd,
        coeff_auc=coeff_auc,
        coeff_soft=coeff_soft,
        device=device,
        verbose=verbose
    )

    # Pop src validation log values
    src_val_loss = src_val_logs["loss"]
    src_val_loss_cls = src_val_logs["loss_cls"]
    src_val_loss_mmd = src_val_logs["loss_mmd"]
    src_val_loss_auc = src_val_logs["loss_auc"]
    src_val_loss_soft = src_val_logs["loss_soft"]
    src_val_acc_raw = src_val_logs["acc_raw"]
    src_val_acc_per_class = src_val_logs["acc_per_class"]
    src_acc_balanced = src_val_logs["acc_balanced"]
    src_probs = src_val_logs["probs"]
    src_preds = src_val_logs["preds"]
    src_labels = src_val_logs["labels"]

    # Pop tgt validation log values
    tgt_val_loss = tgt_val_logs["loss"]
    tgt_val_loss_cls = tgt_val_logs["loss_cls"]
    tgt_val_loss_mmd = tgt_val_logs["loss_mmd"]
    tgt_val_loss_auc = tgt_val_logs["loss_auc"]
    tgt_val_loss_soft = tgt_val_logs["loss_soft"]
    tgt_val_acc_raw = tgt_val_logs["acc_raw"]
    tgt_val_acc_per_class = tgt_val_logs["acc_per_class"]
    tgt_acc_balanced = tgt_val_logs["acc_balanced"]
    tgt_probs = tgt_val_logs["probs"]
    tgt_preds = tgt_val_logs["preds"]
    tgt_labels = tgt_val_logs["labels"]

    # Create figure
    fig, axs = plt.subplots(2, 3, figsize=metrics_plot_figsize)

    # Plot loss coefficients
    alpha_values       = [alpha_fn(e, nepochs) if callable(alpha_fn) else alpha_fn for e in range(nepochs)]
    lambda_values      = [lambda_fn(e, nepochs) if callable(lambda_fn) else lambda_fn for e in range(nepochs)]
    loss_coeffs        = [alpha_values, lambda_values]
    loss_coeffs_kwargs = [{'label':'alpha'},{'label':'lambda'}]
    plot_epoch_metrics(
        axs[0,1],
        nepochs,
        title='Loss Coefficients',
        xlabel='Epoch',
        ylabel='Loss Coefficient',
        yscale=None,
        xscale=None,
        legend_bbox_to_anchor=None,
        legend_loc='best',
        epoch_metrics=loss_coeffs,
        plot_kwargs=loss_coeffs_kwargs,
        normalize_to_max=False
    )

    # Plot learning rate
    lrs        = [train_logs['lrs']]
    lrs_kwargs = [{'label':'lr'}]
    plot_epoch_metrics(
        axs[1,1],
        nepochs,
        title='Learning Rate',
        xlabel='Epoch',
        ylabel='Learning Rate',
        yscale='log',
        xscale=None,
        legend_bbox_to_anchor=None,
        legend_loc='best',
        epoch_metrics=lrs,
        plot_kwargs=lrs_kwargs,
        normalize_to_max=False
    )

    # Plot training and validation losses
    train_losses  = [train_logs[key] for key in train_logs if 'train_loss' in key]
    val_losses    = [train_logs[key] for key in train_logs if 'val_loss' in key]
    losses = [*train_losses, *val_losses]
    train_losses_kwargs  = [{'label':key} for key in train_logs if 'train_loss' in key]
    val_losses_kwargs    = [{'label':key, 'linestyle':':'} for key in train_logs if 'val_loss' in key]
    losses_kwargs = [*train_losses_kwargs, *val_losses_kwargs]
    plot_epoch_metrics(
        axs[0,2],
        nepochs,
        title='Losses',
        xlabel='Epoch',
        ylabel='Loss',
        yscale='log',
        xscale=None,
        legend_bbox_to_anchor=(1.05, 1),
        legend_loc='upper left',
        epoch_metrics=losses,
        plot_kwargs=losses_kwargs,
        normalize_to_max=True
    )

    # Plot training and validation accuracies
    train_accs  = [train_logs[key] for key in train_logs if 'train_acc' in key]
    val_accs    = [train_logs[key] for key in train_logs if 'val_acc' in key]
    print(train_logs.keys())
    print([key for key in train_logs if 'train_acc' in key])
    print([key for key in train_logs if 'val_acc' in key])
    accs = [*train_accs, *val_accs]
    train_accs_kwargs  = [{'label':key} for key in train_logs if 'train_acc' in key]
    val_accs_kwargs    = [{'label':key, 'linestyle':':'} for key in train_logs if 'val_acc' in key]
    accs_kwargs = [*train_accs_kwargs, *val_accs_kwargs]
    plot_epoch_metrics(
        axs[1,2],
        nepochs,
        title='Accuracies',
        xlabel='Epoch',
        ylabel='Accuracy',
        yscale=None,
        xscale=None,
        legend_bbox_to_anchor=(1.05, 1),
        legend_loc='upper left',
        epoch_metrics=accs,
        plot_kwargs=accs_kwargs,
        normalize_to_max=True
    )

    # Plot domain predictions
    plot_domain_preds(axs[0,0], src_probs[:,1], tgt_probs[:,1])

    # Plot ROC AUC curve
    roc_info, thresholds = get_best_threshold(src_labels, src_probs[:,1])
    plot_roc(
        axs[1,0],
        **roc_info
    )

    # Save and show plot
    plt.tight_layout()
    fig.savefig(osp.join(out_dir,metrics_plot_path))

    # Save training logs
    with open(osp.join(out_dir,logs_path), "w") as f:
        json.dump({
            'train':train_logs,
            'src_val':[el if type(el)!=torch.Tensor else el.tolist() for el in src_val_logs],
            'tgt_val_logs':[el if type(el)!=torch.Tensor else el.tolist() for el in src_val_logs]
        }, f, indent=2)

    #----- t-SNE model representation
    src_embeds, src_labels, src_domains, src_preds = collect_embeddings(encoder, clf, src_val_loader_unweighted, device, domain_label=0)
    tgt_embeds, tgt_labels, tgt_domains, tgt_preds = collect_embeddings(encoder, clf, tgt_val_loader, device, domain_label=1)

    # Combine
    all_embeds = torch.cat([src_embeds, tgt_embeds], dim=0)
    all_labels = torch.cat([src_labels, tgt_labels], dim=0)
    all_domains = torch.cat([src_domains, tgt_domains], dim=0)
    all_preds = torch.cat([src_preds, tgt_preds], dim=0)
    labels_and_preds = torch.cat([src_labels, tgt_preds], dim=0)

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=tsne_plot_figsize)

    # Plot
    plot_tsne(axs[0], all_embeds.numpy(), labels_and_preds, all_domains, title='t-SNE representation : true source labels')
    plot_tsne(axs[1], all_embeds.numpy(), all_preds, all_domains, title='t-SNE representation : model predicted labels')

    # Save and show t-SNE fig
    plt.tight_layout()
    fig.savefig(osp.join(out_dir,tsne_plot_path))

    #-----

    # Get kinematics for source and target domains
    src_sg_kin, src_bg_kin = get_kinematics(encoder, clf, src_val_loader_unweighted, threshold=best_thr, device=device,
                                    class_idx_signal=1, class_idx_background=0)
    tgt_sg_kin, tgt_bg_kin = get_kinematics(encoder, clf, tgt_val_loader, threshold=best_thr, device=device,
                                    class_idx_signal=1, class_idx_background=0)

    try:

        # Plot kinematics for source and target domains
        src_fig, src_axs = plot_kinematics(kinematics_axs, src_sg_kin, src_bg_kin, kin_indices=kin_indices, kin_xlabels=kin_xlabels,
                            sg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Signal', 'color':'C0', 'density':True},
                            bg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Background', 'color':'C1', 'density':True}
        )
        tgt_fig, tgt_axs = plot_kinematics(kinematics_axs, tgt_sg_kin, tgt_bg_kin, kin_indices=kin_indices, kin_xlabels=kin_xlabels,
                            sg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Signal', 'color':'C0', 'density':True},
                            bg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Background', 'color':'C1', 'density':True}
        )
    
        # Save and plot kinematics figures
        plt.tight_layout()
        src_fig.savefig(osp.join(out_dir,src_kinematics_plot_path))
        tgt_fig.savefig(osp.join(out_dir,tgt_kinematics_plot_path))

    except ValueError:
        pass

    # Set output paths
    paths = [metrics_plot_path, tsne_plot_path, src_kinematics_plot_path, tgt_kinematics_plot_path]
    paths = [osp.join(out_dir,path) for path in paths]

    return roc_info, src_val_logs, tgt_val_logs, paths

#----------------------------------------------------------------------------------------------------#
# OPTIMIZATION

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback
import wandb
import os
from uuid import uuid4
from pathlib import Path
import sqlite3

# Define the objective function
def objective(trial):
    # Sample hyperparameters
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    # Create a unique output directory for this trial
    trial_id = str(uuid4())
    output_dir = Path("experiments") / trial_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log to wandb
    wandb_run = wandb.init(
        project="hyperparam-optimization",
        name=f"trial-{trial.number}",
        config={"lr": lr, "dropout": dropout, "weight_decay": weight_decay},
        dir=str(output_dir),
        reinit=True,
    )

    try:
        logs = pipeline(
            config_dir=output_dir,
            lr=lr,
            dropout=dropout,
            weight_decay=weight_decay
        )
    except Exception as e:
        wandb_run.finish(exit_code=1)
        raise optuna.exceptions.TrialPruned()  # or fail silently

    # Get the AUC from first log dictionary
    auc = logs[0].get("auc", 0.0)

    # Log metrics to wandb
    wandb_run.log({"auc": auc})
    wandb_run.finish()

    return auc  # Higher is better (maximize)

# SQL-backed Optuna study
storage = optuna.storages.RDBStorage(
    url="sqlite:///optuna_study.db"  # or your PostgreSQL/MySQL URL
)

study = optuna.create_study(
    direction="maximize",
    study_name="model_hpo",
    storage=storage,
    load_if_exists=True,
)

# Optional: use a callback to also log params and scores to WANDB dashboard
wandb_callback = WeightsAndBiasesCallback(metric_name="auc", as_multirun=True)

# Optimize
study.optimize(objective, n_trials=100, callbacks=[wandb_callback])


#----------------------------------------------------------------------------------------------------#
# Script
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import random_split, WeightedRandomSampler
import os.path as osp

DATASET_NAME = 'LAMBDAS'
transform = None #T.Compose([T.ToUndirected(),T.KNNGraph(k=6),T.NormalizeFeatures()]),
max_idx = 1000
src_root='/work/clas12/users/mfmce/pyg_test_rec_particle_dataset_3_7_25/'
tgt_root='/work/clas12/users/mfmce/pyg_DATA_rec_particle_dataset_3_5_24/'

# Load full PROTEINS dataset
full_ds, src_ds, tgt_ds = None, None, None
if DATASET_NAME == 'PROTEINS':
    full_ds = TUDataset(root=osp.expanduser('~/drop/data/'+DATASET_NAME), name=DATASET_NAME)
    
    # Shuffle and split into two subsets
    total_len = len(full_ds)
    split_len = total_len // 2
    src_ds, tgt_ds = random_split(full_ds, [split_len, total_len - split_len])

if DATASET_NAME == 'LAMBDAS':
    
    #----- Load datasets -----#
    src_ds = SmallDataset(
            src_root,
            transform=transform, 
            pre_transform=None,
            pre_filter=None
        )[0:max_idx]
    
    tgt_ds = SmallDataset(
            tgt_root,
            transform=transform,
            pre_transform=None,
            pre_filter=None
        )[0:max_idx]

batch_size = 32
drop_last = True
#----- Create weighted data loader for source and target data -----#

sampler_train_weights = get_sampler_weights(src_train_ds)

sampler_train = WeightedRandomSampler(weights=sampler_train_weights,
                                 num_samples=len(src_train_ds),
                                 replacement=True)

# Create DataLoaders
src_train_loader = DataLoader(src_train_ds, batch_size=batch_size, sampler=sampler_train, drop_last=drop_last)
src_train_loader_unweighted = DataLoader(src_train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
tgt_train_loader = DataLoader(tgt_train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

#----- Create weighted data loader for source and target data -----#

sampler_val_weights = get_sampler_weights(src_val_ds)

sampler_val = WeightedRandomSampler(weights=sampler_val_weights,
                                 num_samples=len(src_val_ds),
                                 replacement=True)

# Create DataLoaders
src_val_loader = DataLoader(src_val_ds, batch_size=batch_size, sampler=sampler_val, drop_last=drop_last)
src_val_loader_unweighted = DataLoader(src_val_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)
tgt_val_loader = DataLoader(tgt_val_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

#--------------------------------------------------------#
# Create model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nepochs = 200
num_classes = 2
gnn_type = "gin"
hdim_gnn = 64
num_layers_gnn = 3
dropout_gnn = 0.4
num_node_features = src_ds[0].num_node_features if DATASET_NAME=="LAMBDAS" else full_ds.num_node_features
heads = 4

num_layers_clf = 3
hdim_clf = 128
dropout_clf = 0.4

hdim_projector = 32

num_layers_dis = 3
hdim_dis = 128
dropout_dis = 0.4

encoder = FlexibleGNNEncoder(
    in_dim=num_node_features,
    hidden_dim=hdim_gnn,
    num_layers=num_layers_gnn,
    gnn_type=gnn_type,      # Try 'gcn', 'sage', 'gat', 'gin'
    dropout=dropout_gnn,
    heads=heads              # Only relevant for GAT
).to(device)

projector = ProjectionHead(hdim_gnn * (heads if gnn_type=="gat" else 1), hdim_projector).to(device)

clf = GraphClassifier(
    in_dim=hdim_gnn * (heads if gnn_type=="gat" else 1),
    out_dim=num_classes,
    num_layers=num_layers_clf,
    hidden_dim=hdim_clf,
    dropout=dropout_clf
).to(device)

disc = DomainDiscriminator(
    in_dim=hdim_gnn * (heads if gnn_type=="gat" else 1),
    num_layers=num_layers_dis,
    hidden_dim=hdim_dis,
    dropout=dropout_dis
).to(device)

#---------- Set optimizer and learning rate scheduler ----------#
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(clf.parameters()) + list(disc.parameters()),
    lr=0.001
)
# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# Linear decay from 1.0 to 0.0 over epochs
lr_lambda = lambda epoch: (1 - (epoch / nepochs))
scheduler = LambdaLR(optimizer, lr_lambda)

num_classes=2
soft_labels_temp=2
nepochs=100
confidence_threshold=0.8
temp_fn=1.0
alpha_fn=1.0
lambda_fn=lambda_fn
coeff_mmd=0.3
coeff_auc=0.01
coeff_soft=0.25
pretrain_frac=0.2
verbose=False

train_logs, soft_labels = train_titok(
    encoder,
    clf,
    src_train_loader,
    tgt_train_loader,
    src_val_loader,
    tgt_val_loader,
    num_classes=num_classes,
    soft_labels_temp=soft_labels_temp,
    nepochs=nepochs,
    confidence_threshold=confidence_threshold,
    temp_fn=temp_fn,
    alpha_fn=alpha_fn,
    lambda_fn=lambda_fn,
    coeff_mmd=coeff_mmd,
    coeff_auc=coeff_auc,
    coeff_soft=coeff_soft,
    pretrain_frac=pretrain_frac,
    device=device,
    verbose=verbose
)

return_labels=True,
pretraining=False,
temp = temp_fn if not callable(temp_fn) else temp_fn(nepochs,nepochs)
alpha = alpha_fn if not callable(alpha_fn) else alpha_fn(nepochs,nepochs)
lambd = lambda_fn if not callable(lambda_fn) else lambda_fn(nepochs,nepochs)
src_val_logs = val_titok(
    encoder,
    clf,
    src_val_loader,
    tgt_val_loader,
    soft_labels,
    return_labels=return_labels,
    pretraining=pretraining,
    num_classes=num_classes,
    confidence_threshold=confidence_threshold,
    temp=temp,
    alpha=alpha,
    lambd=lambd,
    coeff_mmd=coeff_mmd,
    coeff_auc=coeff_auc,
    coeff_soft=coeff_soft,
    device=device,
    verbose=verbose
)

tgt_val_logs = val_titok(
    encoder,
    clf,
    tgt_val_loader,
    tgt_val_loader,
    soft_labels,
    return_labels=return_labels,
    pretraining=pretraining,
    num_classes=num_classes,
    confidence_threshold=confidence_threshold,
    temp=temp,
    alpha=alpha,
    lambd=lambd,
    coeff_mmd=coeff_mmd,
    coeff_auc=coeff_auc,
    coeff_soft=coeff_soft,
    device=device,
    verbose=verbose
)

# Pop src validation log values
src_val_loss = src_val_logs["loss"]
src_val_loss_cls = src_val_logs["loss_cls"]
src_val_loss_mmd = src_val_logs["loss_mmd"]
src_val_loss_auc = src_val_logs["loss_auc"]
src_val_loss_soft = src_val_logs["loss_soft"]
src_val_acc_raw = src_val_logs["acc_raw"]
src_val_acc_per_class = src_val_logs["acc_per_class"]
src_acc_balanced = src_val_logs["acc_balanced"]
src_probs = src_val_logs["probs"]
src_preds = src_val_logs["preds"]
src_labels = src_val_logs["labels"]

# Pop tgt validation log values
tgt_val_loss = tgt_val_logs["loss"]
tgt_val_loss_cls = tgt_val_logs["loss_cls"]
tgt_val_loss_mmd = tgt_val_logs["loss_mmd"]
tgt_val_loss_auc = tgt_val_logs["loss_auc"]
tgt_val_loss_soft = tgt_val_logs["loss_soft"]
tgt_val_acc_raw = tgt_val_logs["acc_raw"]
tgt_val_acc_per_class = tgt_val_logs["acc_per_class"]
tgt_acc_balanced = tgt_val_logs["acc_balanced"]
tgt_probs = tgt_val_logs["probs"]
tgt_preds = tgt_val_logs["preds"]
tgt_labels = tgt_val_logs["labels"]

import json

metrics_plot_path = 'metrics_plot.pdf'
logs_path = 'logs.json'
figsize=(24,12)

# Create figure
fig, axs = plt.subplots(2, 3, figsize=figsize)

# Plot loss coefficients
alpha_values       = [alpha_fn(e, nepochs) if callable(alpha_fn) else alpha_fn for e in range(nepochs)]
lambda_values      = [lambda_fn(e, nepochs) if callable(lambda_fn) else lambda_fn for e in range(nepochs)]
loss_coeffs        = [alpha_values, lambda_values]
loss_coeffs_kwargs = [{'label':'alpha'},{'label':'lambda'}]
plot_epoch_metrics(
    axs[0,1],
    nepochs,
    title='Loss Coefficients',
    xlabel='Epoch',
    ylabel='Loss Coefficient',
    yscale=None,
    xscale=None,
    legend_bbox_to_anchor=None,
    legend_loc='best',
    epoch_metrics=loss_coeffs,
    plot_kwargs=loss_coeffs_kwargs,
    normalize_to_max=False
)

# Plot learning rate
lrs        = [train_logs['lrs']]
lrs_kwargs = [{'label':'lr'}]
plot_epoch_metrics(
    axs[1,1],
    nepochs,
    title='Learning Rate',
    xlabel='Epoch',
    ylabel='Learning Rate',
    yscale='log',
    xscale=None,
    legend_bbox_to_anchor=None,
    legend_loc='best',
    epoch_metrics=lrs,
    plot_kwargs=lrs_kwargs,
    normalize_to_max=False
)

# Plot training and validation losses
train_losses  = [train_logs[key] for key in train_logs if 'train_loss' in key]
val_losses    = [train_logs[key] for key in train_logs if 'val_loss' in key]
losses = [*train_losses, *val_losses]
train_losses_kwargs  = [{'label':key} for key in train_logs if 'train_loss' in key]
val_losses_kwargs    = [{'label':key, 'linestyle':':'} for key in train_logs if 'val_loss' in key]
losses_kwargs = [*train_losses_kwargs, *val_losses_kwargs]
plot_epoch_metrics(
    axs[0,2],
    nepochs,
    title='Losses',
    xlabel='Epoch',
    ylabel='Loss',
    yscale='log',
    xscale=None,
    legend_bbox_to_anchor=(1.05, 1),
    legend_loc='upper left',
    epoch_metrics=losses,
    plot_kwargs=losses_kwargs,
    normalize_to_max=True
)

# Plot training and validation accuracies
train_accs  = [train_logs[key] for key in train_logs if 'train_acc' in key]
val_accs    = [train_logs[key] for key in train_logs if 'val_acc' in key]
print(train_logs.keys())
print([key for key in train_logs if 'train_acc' in key])
print([key for key in train_logs if 'val_acc' in key])
accs = [*train_accs, *val_accs]
train_accs_kwargs  = [{'label':key} for key in train_logs if 'train_acc' in key]
val_accs_kwargs    = [{'label':key, 'linestyle':':'} for key in train_logs if 'val_acc' in key]
accs_kwargs = [*train_accs_kwargs, *val_accs_kwargs]
plot_epoch_metrics(
    axs[1,2],
    nepochs,
    title='Accuracies',
    xlabel='Epoch',
    ylabel='Accuracy',
    yscale=None,
    xscale=None,
    legend_bbox_to_anchor=(1.05, 1),
    legend_loc='upper left',
    epoch_metrics=accs,
    plot_kwargs=accs_kwargs,
    normalize_to_max=True
)

# Plot domain predictions
plot_domain_preds(axs[0,0], src_probs[:,1], tgt_probs[:,1])

# Plot ROC AUC curve
roc_info, thresholds = get_best_threshold(src_labels, src_probs[:,1])
plot_roc(
    axs[1,0],
    **roc_info
)

# Save and show plot
plt.tight_layout()
fig.savefig(metrics_plot_path)
plt.show()

# Save training logs
with open(logs_path, "w") as f:
    json.dump({
        'train':train_logs,
        'src_val':[el if type(el)!=torch.Tensor else el.tolist() for el in src_val_logs],
        'tgt_val_logs':[el if type(el)!=torch.Tensor else el.tolist() for el in src_val_logs]
    }, f, indent=2)

src_embeds, src_labels, src_domains, src_preds = collect_embeddings(encoder, clf, src_val_loader_unweighted, device, domain_label=0)
tgt_embeds, tgt_labels, tgt_domains, tgt_preds = collect_embeddings(encoder, clf, tgt_val_loader, device, domain_label=1)

# Combine
all_embeds = torch.cat([src_embeds, tgt_embeds], dim=0)
all_labels = torch.cat([src_labels, tgt_labels], dim=0)
all_domains = torch.cat([src_domains, tgt_domains], dim=0)
all_preds = torch.cat([src_preds, tgt_preds], dim=0)
labels_and_preds = torch.cat([src_labels, tgt_preds], dim=0)


# Create plot
tsne_plot_path = 'tsne_plot.pdf'
figsize=(20,8)

# Create figure
fig, axs = plt.subplots(1, 2, figsize=figsize)

# Plot
plot_tsne(axs[0], all_embeds.numpy(), labels_and_preds, all_domains, title='t-SNE representation : true source labels')
plot_tsne(axs[1], all_embeds.numpy(), all_preds, all_domains, title='t-SNE representation : model predicted labels')

# Save and show fig
plt.tight_layout()
fig.savefig(tsne_plot_path)
plt.show()

# Plot kinematics arguments
kin_indices = [i for i in range(3,11)]
kin_xlabels = ['$Q^2$ (GeV$^2$)', '$\\nu$', '$W$ (GeV)', '$x$', '$y$', '$z_{p\\pi^{-}}$', '$x_{F p\\pi^{-}}$', '$M_{p\\pi^{-}}$ (GeV)'] # 'idxe', 'idxp', 'idxpi', 
best_thr = roc_info['best_thr']
src_kinematics_plot_path = 'src_kinematics_plot.pdf'
tgt_kinematics_plot_path = 'tgt_kinematics_plot.pdf'

axs = None

# Get kinematics for source and target domains
src_sg_kin, src_bg_kin = get_kinematics(encoder, clf, src_val_loader_unweighted, threshold=best_thr, device=device,
                                  class_idx_signal=1, class_idx_background=0)
tgt_sg_kin, tgt_bg_kin = get_kinematics(encoder, clf, tgt_val_loader, threshold=best_thr, device=device,
                                  class_idx_signal=1, class_idx_background=0)

print(len(kin_indices))
print(len(kin_xlabels))

# Plot kinematics for source and target domains
src_fig, src_axs = plot_kinematics(axs, src_sg_kin, src_bg_kin, kin_indices=kin_indices, kin_xlabels=kin_xlabels,
                    sg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Signal', 'color':'C0', 'density':True},
                    bg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Background', 'color':'C1', 'density':True}
)
tgt_fig, tgt_axs = plot_kinematics(axs, tgt_sg_kin, tgt_bg_kin, kin_indices=kin_indices, kin_xlabels=kin_xlabels,
                    sg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Signal', 'color':'C0', 'density':True},
                    bg_hist_kwargs={'bins':50, 'alpha':0.6, 'label':'Background', 'color':'C1', 'density':True}
)

# Save and plot figures
plt.tight_layout()
fig.savefig(src_kinematics_plot_path)
fig.savefig(tgt_kinematics_plot_path)
plt.show()
