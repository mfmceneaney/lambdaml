#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

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
    :description: Evaluate a regression model on a dataset and return total loss, output values, ground truth labels and optionally kinematics.

    :param: model
    :param: dataloader
    :param: criterion
    :param: get_kins

    :return: loss_tot, outs, ys, kins
    """

    # Put the model in inference mode
    model.eval()

    # Set overall metrics
    loss  = 0
    outs  = []
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
        
        # Add evaluation data to overall lists
        outs.extend(out.cpu())
        ys.extend(data.y.cpu())
        if get_kins: kins.extend(data.kin.cpu())

    return loss, outs, ys, kins

@torch.no_grad()
def test_nolabels(
            model,
            dataloader,
            get_kins = True
        ):
    """
    :description: Evaluate a regression model on a dataset and return output values and kinematics.

    :param: model
    :param: dataloader
    :param: get_kins

    :return: outs, kins
    """

    # Put the model in inference mode
    model.eval()

    # Set overall metrics
    outs  = []
    kins  = []

    # Loop dataloader
    for data in dataloader:

        # Copy data to model device
        data = data.to(model.device)

        # Apply model to data
        out = model(data.x, data.edge_index, data.batch)
        
        # Add evaluation data to overall lists
        outs.extend(out.cpu())
        if get_kins: kins.extend(data.kin.cpu())

    return outs, kins

#TODO: Write method to plot differences and fractional differences from predicted values

def get_regression_metrics(
                            outs,
                            ys,
                            kins = None,
                            get_plots = False,
                        ):
    """
    :description: Get metrics and optionally plots describing regression performance.

    :param: outs
    :param: ys
    :param: get_plots

    :return: resolutions, plots
    """

    #TODO: Compute resolutions
    resolutions = {}

    # Get plots if
    plots = {}
    if get_plots:
        pass#TODO: WRITE THIS!

    return resolutions, plots