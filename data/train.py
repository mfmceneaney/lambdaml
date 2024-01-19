#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Define training and testing routines
def train(
            model,
            dataloader,
            optimizer,
            criterion
        ):
    """
    :description: Train a model for one epoch on a dataset.

    :param: model
    :param: dataloader
    :param: optimizer
    :param: criterion
    """

    # Put model in training mode
    model.train()

    # Loop dataloader
    for data in dataloader:

        # Copy data to device
        data = data.to(model.device)

        # Zero gradients
        optimizer.zero_grad()  # Clear gradients.

        # Apply model to data
        out = model(data.x, data.edge_index, data.batch)

        # Compute loss
        loss = criterion(out, data.y)

        # Compute gradients and optimize parameters
        loss.backward()
        optimizer.step()
        

@torch.no_grad()
def test(
            model,
            dataloader,
            criterion
        ):
    """
    :description: Evaluate a classification model on a dataset and return total loss, output probabilities, label predictions, and ground truth labels.

    :param: model
    :param: dataloader
    :param: criterion

    :return: loss_tot, outs, preds, ys
    """

    # Put the model in inference mode
    model.eval()

    # Set overall metrics
    loss  = 0
    outs  = []
    preds = []
    ys    = []

    # Loop dataloader
    for data in dataloader:

        # Copy data to model device
        data = data.to(model.device)

        # Apply model to data
        out = model(data.x, data.edge_index, data.batch)

        # Compute losss
        l = criterion(out, data.y)
        loss += l.item()/len(dataloader.dataset)

        # Compute probability and make prediction
        out  = softmax(out,dim=-1)
        pred = out.argmax(dim=1)
        
        # Add evaluation data to overall lists
        outs.extend(out.cpu())
        preds.extend(pred.cpu())
        ys.extend(data.y.cpu())

    return loss, outs, preds, ys

def get_binary_classification_metrics(
                                        preds,
                                        ys
                                    ):
    """
    :description: Get metrics and plots describing binary classification performance.

    :param: loss
    :param: outs
    :param: preds
    :param: ys

    :return: accuracy, precision, recall, precision_n, recall_n, roc_auc
    """

    # Get confiusion matrix
    cm = confusion_matrix(ys,preds,labels=[0,1])
    tp = cm[1,1]
    fp = cm[0,1] # bg classified as sig is 0,1
    fn = cm[1,0] # signal classified as bg is 1,0
    tn = cm[0,0]
        
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) # accuracy of the identified signal events
    recall = tp / (tp + fn) # efficiency
    precision_n = tn / (tn + fn)
    recall_n = tn / (tn + fp)
    
    # Compute roc auc
    roc_auc = roc_auc_score(ys,preds)

    # Compute roc curve
    roc_c = roc_curve(ys,preds)

    return accuracy, precision, recall, precision_n, recall_n, roc_auc, roc_c
