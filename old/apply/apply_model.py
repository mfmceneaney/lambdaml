#----------------------------------------------------------------------#
# Lambda event classification model application script.
# NOTE: This script will add a bank with model prediction to an input
# HIPO file.
#----------------------------------------------------------------------#

# Data
import numpy as np
import hipopy.hipopy as hp

# Miscellaneous
import os
import sys #NOTE: ADDED
from tqdm import tqdm
import argparse

# ML
import torch
from torch_geometric.data import Data

# Local
from preprocessing import get_event_table, preprocess_rec_traj, get_edge_index_rec_traj

def main(
        modelpath = "model.pt",
        filename  = "test.hipo", # Recreate this in your $PWD,
        bank      = "ML::pred",
        dtype     = "D", #NOTE: For now all the bank entries have to have the same type.
        names     = ["MLPred","MLLabel"],
    ):

    # Make sure you have absolute paths
    modelpath = os.path.abspath(modelpath)
    filename  = os.path.abspath(filename)

    # Load model and connect to device
    model = torch.load(modelpath, weights_only=False)
    model.eval()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Open file
    namesAndTypes = {e:dtype for e in names}
    file = hp.recreate(filename)
    file.newTree(bank,namesAndTypes)
    file.open() # IMPORTANT!  Open AFTER calling newTree, otherwise the banks will not be written!

    # Loop file
    for event_idx, event in tqdm(enumerate(file)):

        #NOTE: loopting event should work here -> still returns batch...
        
        # Get REC::Traj bank
        rec_traj_event_table = get_event_table(rec_traj_keys,event_num,batch,dtype=float)

        # Preprocess data from REC::Traj instead #NOTE: This also automatically selects subarray
        x = preprocess_rec_traj(rec_traj_event_table)
        x = torch.tensor(x,dtype=torch.float32)
        if torch.abs(x).max()>1.0:
            print("DEBUGGING: torch.where(torch.abs(x)>1.0) = ",torch.where(torch.abs(x)>1.0))
            print("DEBUGGING: torch.abs(x).max()            = ",torch.abs(x).max())
        
        # Define edge index
        edge_index = get_edge_index_rec_traj(rec_traj_event_table)
        
        # Create PyG graph
        data = Data(x=x, edge_index=edge_index.t().contiguous())

        # Apply model
        bankdata = None
        with torch.no_grad():
            data = data.to(device)
            out  = model(data.x, data.edge_index, data.batch)
            out  = torch.nn.functional.softmax(out,dim=-1)[1] #NOTE: GIVE PROBABILITY OF SIGNAL CLASS AT INDEX 1.
            pred = out.argmax(dim=1).item()

            # Set bank data
            bankdata = np.array([[out,pred]])

        # Add data to even events
        if bankdata is not None: file.update({bank : bankdata})
        else: file.update({}) #NOTE: Important to write empty events too!

    # Close file
    file.close()

#---------- Main ----------#
if __name__=="__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description='Apply a PyTorch binary clasification model to a HIPO file and output results to a new HIPO bank in the same file.')
    parser.add_argument('--modelpath', type=str, default='model.pt',help='Path to PyTorch model')
    parser.add_argument('--hipofile',type=str, default='test.hipo',help='Path to input HIPO file')
    parser.add_argument('--bank', type=str, default='ML::pred',help='HIPO bank name to which to write ML results')
    parser.add_argument('--dtype',type=str, default="D",help='HIPO bank data type ("D" -> double, "F" -> float etc.)')
    parser.add_argument('--names',type=str, default=["MLPred","MLLabel"], nargs=2, help='HIPO bank row names')
    args = parser.parse_args()

    # Run main
    main(
        modelpath = args.modelpath,
        filename  = args.hipofile,
        bank      = args.bank,
        dtype     = args.dtype,
        names     = args.names,
    )
