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
            criterion,
            get_kins = False
        ):
    """
    :description: Evaluate a classification model on a dataset and return total loss, output probabilities, label predictions, and ground truth labels and optionally kinematics.

    :param: model
    :param: dataloader
    :param: criterion
    :param: get_kins

    :return: loss_tot, outs, preds, ys, kins
    """

    # Put the model in inference mode
    model.eval()

    # Set overall metrics
    loss  = 0
    outs  = []
    preds = []
    ys    = []
    kins  = []

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
        if get_kins: kins.extend(data.kin.cpu())

    return loss, outs, preds, ys, kins

@torch.no_grad()
def test_nolabels(
            model,
            dataloader,
            get_kins = True
        ):
    """
    :description: Evaluate a classification model on a dataset and return output probabilities, label predictions, and kinematics.

    :param: model
    :param: dataloader
    :param: get_kins

    :return: outs, preds, kins
    """

    # Put the model in inference mode
    model.eval()

    # Set overall metrics
    outs  = []
    preds = []
    kins  = []

    # Loop dataloader
    for data in dataloader:

        # Copy data to model device
        data = data.to(model.device)

        # Apply model to data
        out = model(data.x, data.edge_index, data.batch)

        # Compute probability and make prediction
        out  = softmax(out,dim=-1)
        pred = out.argmax(dim=1)
        
        # Add evaluation data to overall lists
        outs.extend(out.cpu())
        preds.extend(pred.cpu())
        if get_kins: kins.extend(data.kin.cpu())

    return outs, preds, kins

def get_binary_classification_metrics(
                                        preds,
                                        ys,
                                        kins = None,
                                        get_plots = False,
                                    ):
    """
    :description: Get metrics and optionally plots describing binary classification performance.

    :param: preds
    :param: ys
    :param: get_plots

    :return: accuracy, precision, recall, precision_n, recall_n, roc_auc, (plots)
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

    # Get plots if
    plots = {}
    if get_plots:
        
        # Plot heatmap of confusion matrix
        classes = ['bg', 'sig']
        title = "Confusion Matrix"
        confusion_matrix_plot = sns.heatmap(cm, cmap="Blues", annot=True, xticklabels=classes, yticklabels=classes, cbar=False)
        confusion_matrix_plot.set(title=title, xlabel="Predicted label", ylabel="True label")
        plots['confusion_matrix_plot'] = confusion_matrix_plot

        # Plot outputs of model
        outputs_sig_true  = outputs[:,1][np.logical_and(decisions==1,y_true==1)]
outputs_sig_false = outputs[:,1][np.logical_and(decisions==1,y_true==0)]
outputs_bg_false  = outputs[:,1][np.logical_and(decisions==0,y_true==1)]
outputs_bg_true   = outputs[:,1][np.logical_and(decisions==0,y_true==0)]

        def plot_data_separated(array_sig_true,array_sig_false,array_bg_false,array_bg_true,title=None,xlabel='index',nbins=50,low=-1.1,high=1.1,logy=False):
    
    array_sig_true = array_sig_true.flatten()
    array_sig_false = array_sig_false.flatten()
    array_bg_false = array_bg_false.flatten()
    array_bg_true = array_bg_true.flatten()
    
    # Plot SIG ONLY distributions
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.title('Separated distribution MC-matched')
    plt.hist(array_sig_true, color='tab:red', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig true')
    plt.hist(array_sig_false, color='tab:orange', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig false')
#     plt.hist(array_bg_false, color='tab:green', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg false')
#     plt.hist(array_bg_true, color='tab:blue', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg true')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    if logy: plt.yscale('log')
#     f.savefig(xlabel+'_separated_'+todays_date+'.pdf')

    # Plot SIG AND BG distributions
    f = plt.figure()
    if title != None:
        plt.title(title)
    plt.title('Separated distribution MC-matched')
    plt.hist(array_sig_true, color='tab:red', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig true')
    plt.hist(array_sig_false, color='tab:orange', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='sig false')
    plt.hist(array_bg_false, color='tab:green', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg false')
    plt.hist(array_bg_true, color='tab:blue', alpha=0.5, range=(low,high), bins=nbins, histtype='stepfilled', density=False, label='bg true')
    plt.legend(loc='upper left', frameon=False)
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    if logy: plt.yscale('log')
#     f.savefig(xlabel+'_separated_'+todays_date+'.pdf')
    plt.show()

        # Plot separated kinematics distributions
        if kins is not None:
            


    return accuracy, precision, recall, precision_n, recall_n, roc_auc, roc_c, plots
