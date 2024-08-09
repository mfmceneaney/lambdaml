#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

import torch
from torch.nn.functional import softmax
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import plotly.express as px
import scipy.optimize as opt
from scipy.stats import crystalball
import scipy.integrate as integrate#TODO: ORGANIZE IMPORTS

# Define training and testing routines
def train(
            model,
            device,
            dataloader,
            optimizer,
            criterion
        ):
    """
    :description: Train a model for one epoch on a dataset.

    :param: model
    :param: device
    :param: dataloader
    :param: optimizer
    :param: criterion
    """

    # Put model in training mode
    model.train()

    # Loop dataloader
    for data in dataloader:

        # Copy data to device
        data = data.to(device)

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
            device,
            dataloader,
            criterion,
            get_kins = False
        ):
    """
    :description: Evaluate a classification model on a dataset and return total loss, output probabilities, label predictions, and ground truth labels and optionally kinematics.

    :param: model
    :param: device
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
        data = data.to(device)

        # Apply model to data
        out = model(data.x, data.edge_index, data.batch)

        # Compute losss
        l = criterion(out, data.y)
        loss += l.item()/len(dataloader.dataset)

        # Compute probability and make prediction
        out  = softmax(out,dim=-1)
        pred = out.argmax(dim=1)
        
        # Add evaluation data to overall lists
        outs = out.cpu() if len(outs)==0 else torch.cat((outs,out.cpu()),axis=0)
        preds = pred.cpu() if len(preds)==0 else torch.cat((preds,pred.cpu()),axis=0)
        ys = data.y.cpu() if len(ys)==0 else torch.cat((ys,data.y.cpu()),axis=0)
        if get_kins: kins.extend(data.kinematics)

    return loss, outs, preds, ys, kins

@torch.no_grad()
def test_nolabels(
            model,
            device,
            dataloader,
            get_kins = True
        ):
    """
    :description: Evaluate a classification model on a dataset and return output probabilities, label predictions, and kinematics.

    :param: model
    :param: device
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
        data = data.to(device)

        # Apply model to data
        out = model(data.x, data.edge_index, data.batch)

        # Compute probability and make prediction
        out  = softmax(out,dim=-1)
        pred = out.argmax(dim=1)
        
        # Add evaluation data to overall lists
        outs = out.cpu() if len(outs)==0 else torch.cat((outs,out.cpu()),axis=0)
        preds = pred.cpu() if len(preds)==0 else torch.cat((preds,pred.cpu()),axis=0)
        if get_kins: kins.extend(data.kinematics)

    return outs, preds, kins

def plot_data_separated(
        sg_true,
        sg_false,
        bg_false,
        bg_true,
        title='Separated distributions MC-matched',
        xlabel='index',
        nbins=50,
        low=-1.1,
        high=1.1,
        logy=False,
        figsize=(16,10)
    ):
    """
    :description: Plot separated distributions of only truth-matched signal and truth-matched signal and backgroud and return plots.

    :param: sg_true
    :param: sg_false
    :param: bg_false
    :param: bg_true
    :param: title
    :param: xlabel
    :param: nbins
    :param: low
    :param: high
    :param: logy
    :param: figsize

    :return: f1, f2
    """

    # Set plotting parameters
    label1 = 'True Signal'
    label2 = 'False Signal'
    c1 = [label1 for i in range(len(sg_true))]
    c2 = [label2 for i in range(len(sg_false))]
    opacity = 0.5

    # Plot histograms
    df = {
        'x':[*sg_true,*sg_false],
        'Distribution':[*c1,*c2]
    }
    labels={'x':xlabel}
    f1 = px.histogram(df, x='x',nbins=nbins ,color='Distribution', title=title, labels=labels, opacity=0.5, range_x=(low,high), barmode="overlay", log_y=logy)

    # Set plotting parameters
    label1 = 'True Signal'
    label2 = 'False Signal'
    label3 = 'False Background'
    label4 = 'True Background'
    c1 = [label1 for i in range(len(sg_true))]
    c2 = [label2 for i in range(len(sg_false))]
    c3 = [label3 for i in range(len(bg_false))]
    c4 = [label4 for i in range(len(bg_true))]
    opacity = 0.5

    # Plot histograms
    df = {
        'x':[*sg_true,*sg_false,*bg_false,*bg_true],
        'Distribution':[*c1,*c2,*c3,*c4]
    }
    labels={'x':xlabel}
    f2 = px.histogram(df, x='x',nbins=nbins ,color='Distribution', title=title, labels=labels, opacity=0.5, range_x=(low,high), barmode="overlay", log_y=logy)

    return f1, f2

def plot_data_sg_bg(
        array_sg,
        array_bg,
        title='Separated signal and background distributions',
        xlabel='index',
        nbins=50,
        low=-1.1,
        high=1.1,
        logy=False,
        figsize=(16,10)
    ):
    """
    :description: Plot separated distributions of only truth-matched signal and truth-matched signal and backgroud and return plots.

    :param: sg
    :param: bg
    :param: title
    :param: xlabel
    :param: nbins
    :param: low
    :param: high
    :param: logy
    :param: figsize

    :return: f1
    """

    # Set plotting parameters
    label1 = 'Signal'
    label2 = 'Background'
    c1 = [label1 for i in range(len(array_sg))]
    c2 = [label2 for i in range(len(array_bg))]
    opacity = 0.5

    # Plot histograms
    df = {
        'x':[*array_sg,*array_bg],
        'Distribution':[*c1,*c2]
    }
    labels={'x':xlabel}
    f1 = px.histogram(df, x='x',nbins=nbins ,color='Distribution', title=title, labels=labels, opacity=0.5, range_x=(low,high), barmode="overlay", log_y=logy)

    return f1

def get_binary_classification_metrics(
                                        outs,
                                        preds,
                                        ys,
                                        kins = None,
                                        get_plots = False,
                                        kin_names = None,
                                        kin_labels = None,
                                    ):
    """
    :description: Get metrics and optionally plots describing binary classification performance.

    :param: outs
    :param: preds
    :param: ys
    :param: kins
    :param: get_plots
    :param: kin_names
    :param: kin_labels

    :return: accuracy, precision, recall, precision_n, recall_n, roc_auc, plots
    """

    # Get confusion matrix
    cm = confusion_matrix(ys,preds,labels=[0,1])
    tp = cm[1,1]
    fp = cm[0,1] # bg classified as sig is 0,1
    fn = cm[1,0] # signal classified as bg is 1,0
    tn = cm[0,0]
    
    # Compute metrics
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) # accuracy of the identified signal events
    recall = tp / (tp + fn) # efficiency
    precision_n = tn / (tn + fn)
    recall_n = tn / (tn + fp)
    
    # Compute roc auc
    roc_auc = roc_auc_score(ys,preds)

    # Compute roc curve
    roc_c = roc_curve(ys,preds)

    # Get plots if requested
    plots = {}
    if get_plots:
        
        # Plot heatmap of confusion matrix
        classes = ['bg', 'sig']
        title = "Confusion Matrix"
        confusion_matrix_plot = px.imshow(
                                        cm, 
                                        labels=dict(x="Predicted label", y="True label", color="Number"),
                                        text_auto=True
                                )
        confusion_matrix_plot.update_xaxes(side="bottom")
        plots['confusion_matrix_plot'] = confusion_matrix_plot

        # Separate outputs of model
        outs_sg_true  = outs[:,1][torch.logical_and(preds==1,ys==1)]
        outs_sg_false = outs[:,1][torch.logical_and(preds==1,ys==0)]
        outs_bg_false  = outs[:,1][torch.logical_and(preds==0,ys==1)]
        outs_bg_true   = outs[:,1][torch.logical_and(preds==0,ys==0)]

        # Plot separated output distributions
        outs_sg, outs_sg_and_bg = plot_data_separated(
            outs_sg_true,
            outs_sg_false,
            outs_bg_false,
            outs_bg_true,
            title='Separated output distributions',
            xlabel='NN output',
            nbins=50,
            low=-1.1,
            high=1.1,
            logy=True,
            figsize=(16,10)
            )
        plots['outs_sg'] = outs_sg
        plots['outs_sg_and_bg'] = outs_sg_and_bg

        # Plot separated kinematics distributions
        if kins is not None:

            # Create awkward array since kinematics will not be same length for every event
            kins = ak.Array(kins)

            # Get truth-matched and flattened kinematics arrays
            kins_sg_true  = kins[np.logical_and(preds.numpy()==1,ys.numpy()==1)] #NOTE: Make sure single match entries for kinematics are wrapped in extra 1D layer!
            kins_sg_false = kins[np.logical_and(preds.numpy()==1,ys.numpy()==0)]
            kins_bg_false = kins[np.logical_and(preds.numpy()==0,ys.numpy()==1)]
            kins_bg_true  = kins[np.logical_and(preds.numpy()==0,ys.numpy()==0)]

            def myflatten(some_array):
                new_array = []
                for el in some_array:
                    if type(el[0])==ak.Array or type(el[0])==list:
                        for sub_el in el:
                            new_array.append(sub_el)
                    else:
                        new_array.append(el)
                return new_array

            # Flatten events with more than one set of kinematics and convert to numpy arrays
            kins          = np.array(myflatten(kins))
            kins_sg_true  = np.array(myflatten(kins_sg_true))
            kins_sg_false = np.array(myflatten(kins_sg_false))
            kins_bg_false = np.array(myflatten(kins_bg_false))
            kins_bg_true  = np.array(myflatten(kins_bg_true))

            # Check dimensions
            if kin_labels is None or len(kin_labels)!=len(kins[0]):
                raise ValueError("`kin_labels` must be set and have the same length as the 2nd dimension of `kins`.")
            if kin_names is None or len(kin_names)!=len(kins[0]):
                raise ValueError("`kin_names` must be set and have the same length as the 2nd dimension of `kins`.")

            # Loop kinematics
            for idx in range(len(kins[0])):
                kin_sg_true  = kins_sg_true[:,idx]
                kin_sg_false = kins_sg_false[:,idx]
                kin_bg_false = kins_bg_false[:,idx]
                kin_bg_true  = kins_bg_true[:,idx]
                kin_label    = kin_labels[idx]
                kin_name     = kin_names[idx]

                # Plot separated kinematic distributions
                kin_sg, kin_sg_and_bg = plot_data_separated(
                    kin_sg_true,
                    kin_sg_false,
                    kin_bg_false,
                    kin_bg_true,
                    title='Separated output distributions',
                    xlabel=kin_label,
                    nbins=50,
                    low=np.min(kins[:,idx]), #TODO: Pass these from config above
                    high=np.max(kins[:,idx]),
                    logy=True,
                    figsize=(16,10)
                    )
                plots[kin_name+'_sg'] = kin_sg
                plots[kin_name+'_sg_and_bg'] = kin_sg_and_bg

    return accuracy, precision, recall, precision_n, recall_n, roc_auc, roc_c, plots

def get_binary_classification_metrics_nolabels(
                                        outs,
                                        preds,
                                        metric_fns = {},
                                        kins = None,
                                        get_plots = False,
                                        kin_names = None,
                                        kin_labels = None,
                                    ):
    """
    :description: Get metrics and optionally plots describing binary classification performance assuming no labels.

    :param: outs
    :param: preds
    :param: metric_fns
    :param: get_plots
    :param: kin_names
    :param: kin_labels

    :return: metrics, plots
    """

    # Set unique label suffix for all objects returned by this method
    identifier_key = '_nolabels'

    # Compute metrics
    metrics = {key+identifier_key: fn(preds,kins,kin_labels) for key, fn in metric_fns}#TODO: Make some assumptions about output of metric function and unpack return entries?

    # Get plots if requested
    plots = {}
    if get_plots:

        # Separate outputs of model
        outs_sg = outs[:,1][preds==1]
        outs_bg = outs[:,1][preds==0]

        # Plot separated output distributions
        outs_sg_and_bg = plot_data_sg_bg(
            outs_sg,
            outs_bg,
            title='Separated output signal and background distributions',
            xlabel='NN output',
            nbins=50,
            low=-1.1,
            high=1.1,
            logy=True,
            figsize=(16,10)
            )
        plots['outs_sg_and_bg'+identifier_key] = outs_sg_and_bg

        # Plot separated kinematics distributions
        if kins is not None:

            # Create awkward array so you can easily flatten later
            kins = ak.Array(kins)

            # Get truth-matched and flattened kinematics arrays
            kins_sg = kins[preds.numpy()==1] #NOTE: Make sure single match entries for kinematics are wrapped in extra 1D layer!
            kins_bg = kins[preds.numpy()==0]

            def myflatten(some_array):
                new_array = []
                for el in some_array:
                    if type(el[0])==ak.Array or type(el[0])==list:
                        for sub_el in el:
                            new_array.append(sub_el)
                    else:
                        new_array.append(el)
                return new_array

            # Flatten events with more than one set of kinematics and convert to numpy arrays
            kins    = np.array(myflatten(kins))
            kins_sg = np.array(myflatten(kins_sg))
            kins_bg = np.array(myflatten(kins_bg))

            # Check dimensions
            if kin_labels is None or len(kin_labels)!=len(kins[0]):
                raise ValueError("`kin_labels` must be set and have the same length as the 2nd dimension of `kins`.")
            if kin_names is None or len(kin_names)!=len(kins[0]):
                raise TypeError("`kin_names` must be set and have the same length as the 2nd dimension of `kins`.")

            # Loop kinematics
            for idx in range(len(kins[0])):
                kin_sg = kins_sg[:,idx]
                kin_bg = kins_bg[:,idx]
                kin_label = kin_labels[idx]
                kin_name  = kin_names[idx]

                # Plot separated kinematic distributions
                kin_sg_and_bg = plot_data_sg_bg(
                    kin_sg,
                    kin_bg,
                    title='Separated output signal and background distributions',
                    xlabel=kin_label,
                    nbins=50,
                    low=np.min(kins[:,idx]), #TODO: Pass these from config above
                    high=np.max(kins[:,idx]),
                    logy=True,
                    figsize=(16,10)
                    )
                plots[kin_name+'_sg_and_bg'+identifier_key] = kin_sg_and_bg

    return metrics, plots

def get_lambda_mass_fit(
        kins,
        true_labels=None,
        sg_min=1.11,
        sg_max=1.13,
        bins=100,
        low_high=(1.08,1.24),
        mass_index=-1,
    ):
    """
    :description: Fit $\Lambda$ hyperon mass spectrum from CLAS12 data and return dictionary of fit output metrics.

    :param: kins
    :param: true_labels
    :param: sg_min
    :param: sg_max
    :param: bins
    :param: low_high
    :param: mass_index

    :return: metrics
    """

    def myflatten(some_array):
        new_array = []
        for el in some_array:
            if type(el[0])==torch.tensor or type(el[0])==list:
                for sub_el in el:
                    new_array.append(sub_el.cpu().tolist())
            else:
                new_array.append(el.cpu().tolist())
        return new_array

    # Flatten events with more than one set of kinematics and convert to numpy arrays
    kins = torch.tensor(myflatten(kins))

    # Create fit functions
    def func(x, N, beta, m, loc, scale, A, B, C):
        return N*crystalball.pdf(-x, beta, m, -loc, scale) + A*(1 - B*(x - C)**2)
        
    def sig(x, N, beta, m, loc, scale):
        return N*crystalball.pdf(-x, beta, m, -loc, scale)
        
    def bg(x, A, B, C):
        return A*(1 - B*(x - C)**2)

    # Create histogram
    f = plt.figure(figsize=(16,10))
    plt.title('Separated mass distribution')
    masses =  kins[:,mass_index]
    hdata = plt.hist(masses, color='tab:blue', alpha=0.5, range=low_high, bins=bins, histtype='stepfilled', density=False, label='signal')

    # Fit histogram
    N, beta, m, loc, scale, A, B, C = 5, 1, 1.112, 1.115, 0.008, hdata[0][-1], 37, 1.24
    if A==0: A = 0.1
    d_N, d_beta, d_m, d_loc, d_scale, d_A, d_B, d_C = N/0.01, beta/0.1, m/0.1, loc/0.1, scale/0.01, A/10, B/0.1, C/0.1
    parsMin = [N-d_N, beta-d_beta, m-d_m, loc-d_loc, scale-d_scale, A-d_A, B-d_B, C-d_C]
    parsMax = [N+d_N, beta+d_beta, m+d_m, loc+d_loc, scale+d_scale, A+d_A, B+d_B, C+d_C]
    optParams, pcov = opt.curve_fit(func, hdata[1][:-1], hdata[0], method='trf', bounds=(parsMin,parsMax))

    # Plot fit
    x = np.linspace(low_high[0],low_high[1],bins)#mass_sig_Y[~mass_sig_Y.mask]
    y = hdata[0]
    plt.plot(x, func(x, *optParams), color='r')
    plt.plot(x, sig(x, *optParams[0:5]), color='tab:purple')
    plt.plot(x, bg(x, *optParams[5:]), color='b')
    bghist = plt.hist(x, weights=y-bg(x, *optParams[5:]), bins=bins, range=low_high, histtype='step', alpha=0.5, color='b')

    # Compute fit chi2/ndf value
    r = np.divide(y - func(x, *optParams),np.sqrt([el if el>0 else 1 for el in func(x, *optParams)]))
    chi2 = np.sum(np.square(r))
    chi2ndf = chi2/(len(y)-len(optParams))

    # Compute S, B, N values from functional integration
    binwidth = (low_high[1]-low_high[0])/bins
    resultN = integrate.quad(lambda x: func(x, *optParams),sg_min,sg_max)[0] / binwidth
    resultS = integrate.quad(lambda x: sig(x, *optParams[0:5]),sg_min,sg_max)[0] / binwidth
    resultB = integrate.quad(lambda x: bg(x, *optParams[5:]),sg_min,sg_max)[0] / binwidth

    # Compute B, N, and FOM, purity from histogram integration
    bin1 = int((sg_min-low_high[0])/binwidth)
    bin2 = int((sg_max-low_high[0])/binwidth)
    integral_bghist = np.sum(bghist[0][bin1:bin2])
    integral_tothist = np.sum(hdata[0][bin1:bin2])
    fom = (integral_tothist-integral_bghist)/np.sqrt(integral_tothist)
    purity = (integral_tothist-integral_bghist)/integral_tothist
    
    # Create legend and set axis labels
    lg = "Fit Info\n-------------------------\n"
    lg += f"N = {round(optParams[0],0)}±{round(pcov[0,0],5)}\n"
    lg += f"α = {round(optParams[1],3)}±{round(pcov[1,1],5)}\n"
    lg += f"n = {round(optParams[2],3)}±{round(pcov[2,2],5)}\n"
    lg += f"μ = {round(optParams[3],5)}±{round(pcov[3,3],5)}\n"
    lg += f"σ = {round(optParams[4],5)}±{round(pcov[4,4],5)}\n"
    lg += f"A = {round(optParams[5],0)}±{round(pcov[5,6],5)}\n"
    lg += f"β = {round(optParams[6],0)}±{round(pcov[6,6],5)}\n"
    lg += f"M = {round(optParams[7],2)}±{round(pcov[7,7],5)}\n"
    plt.text(low_high[1]-(low_high[1]-low_high[0])/3,2/3*max(hdata[0]),lg,fontsize=16,linespacing=1.5)
    plt.ylabel('Counts')
    plt.xlabel('Invariant mass (GeV)')

    # Compute true S, B, N, FOM, purity values if truth is given
    if true_labels is not None:
        pass#TODO! Compute and plot true signal hist???

    return { #TODO: Return f, resultsN, resultS, resultB, integral_bghist, integral_tothist, fom, purity, ... true values in signal region
        'massfit_plot':f,
        'resultN':resultN,
        'resultS':resultS,
        'resultB':resultB,
        'integral_bghist':integral_bghist,
        'integral_tothist':integral_tothist,
        'fom':fom,
        'purity':purity,
    }
