#----------------------------------------------------------------------#
# Author: Matthew McEneaney, Duke University
#----------------------------------------------------------------------#

import argparse
import hipopy.hipopy as hp
import torch
import torch_geometric as tg

# Local imports
from data import CustomInMemoryDataset
from processing import *

def main(infiles,outdir,step,start=0,nolabels=False):
    """
    :description: Create PyG dataset and save to file.  Graph data is taken from REC::Traj and preprocessed with processing.preprocess_rec_traj.

    :param: infiles
    :param: step
    :param: outdir
    :param: start
    :param: nolabels
    """

    #----------------------------------------------------------------------#
    # File iteration

    # Set banks, and step size
    banks = [
        'REC::Particle',
        'REC::Traj',
        'REC::Kinematics',
        'MC::Lund',
    ]
    if not nolabels: banks.append('MC::Lund')

    # Iterate hipo files
    datalist = []
    for batch_num, batch in tqdm.tqdm(enumerate(hp.iterate(infiles,banks=banks,step=step))):
        
        # Set bank names and entry names to look at
        all_keys            = list(batch.keys())
        rec_particle_name   = 'REC::Particle'
        rec_particle_keys   = get_bank_keys(rec_particle_name,all_keys)
        rec_traj_name       = 'REC::Traj'
        rec_traj_keys       = get_bank_keys(rec_traj_name,all_keys)
        rec_kinematics_name = 'REC::Kinematics'
        rec_kinematics_keys = get_bank_keys(rec_kinematics_name,all_keys)
        mc_lund_name, mc_lund_keys = None, None
        if not nolabels:
            mc_lund_name        = 'MC::Lund'
            mc_lund_keys        = get_bank_keys(mc_lund_name,all_keys)
        
        # Loop events in batch
        for event_num, _ in enumerate(range(0,len(batch[list(batch.keys())[0]]))):
            
            # Check scattered electron and proton pion in REC::Particle #NOTE: Should prefilter events anyway but check just in case.
            filter_pids  = [2212,-211]
            rec_pid_idx  = 0
            no_filter_pids = True
            count_filter_pids = 0
            for pid in filter_pids:
                if pid in batch['REC::Particle_pid'][event_num]: count_filter_pids += 1
            if count_filter_pids == len(filter_pids): no_filter_pids = False
            if batch['REC::Particle_pid'][event_num][0]!=11 or no_filter_pids: continue #NOTE: Check that particles of interest actually present in event
            
            # Get REC::Particle bank
            rec_particle_event_table = get_event_table(rec_particle_keys,event_num,batch,dtype=float)
            
            # Get REC::Traj bank
            rec_traj_event_table = get_event_table(rec_traj_keys,event_num,batch,dtype=float)
            
            # Get REC::Kinematics bank
            rec_kinematics_event_table = get_event_table(rec_kinematics_keys,event_num,batch,dtype=float)
            
            # Get MC::Lund bank and MC->REC matching indices
            y = [0]
            if not nolabels:
                mc_lund_event_table = get_event_table(mc_lund_keys,event_num,batch,dtype=float)
                match_indices  = get_match_indices(rec_particle_event_table,mc_lund_event_table)#TODO: This is somehow resetting rec_particle_event_table...
                
                # Check MC::Lund matched indices for Lambda decay and set event label accordingly
                has_decay, rec_indices = check_has_decay(rec_particle_event_table,mc_lund_event_table,match_indices,decay,
                                        rec_particle_pid_idx=0,mc_lund_pid_idx=3,mc_lund_parent_idx=4,mc_lund_daughter_idx=5)#NOTE: TODO: Decay should be formatted as follows: [parent_pid,[daughter_pid1,daughter_pid2,...]]
                y = [1] if has_decay==True else [0]

            # Preprocess data from REC::Traj instead #NOTE: This also automatically selects subarray
            x = preprocess_rec_traj(rec_traj_event_table)
            x = torch.tensor(x,dtype=torch.float32)
            
            # Define edge index
            edge_index = get_edge_index_rec_traj(rec_traj_event_table) #NOTE: NOW USE PARTICLE TRACK LINKING FROM HIPO BANK. 1/17/24
            
            # Create PyG graph
            data             = tg.data.Data(x=x, edge_index=edge_index.t().contiguous())
            data.y           = torch.tensor(y,dtype=torch.long) #NOTE: Add extra dimension here so that training gets target batch dimension right.
            data.kinematics  = torch.tensor(rec_kinematics_event_table,dtype=torch.float32)
            data.rec_indices = torch.tensor(rec_indices,dtype=torch.long)
            
            # Add graph to dataset
            datalist.append(data)

    # Write datalist to PyG Dataset
    dataset = CustomInMemoryDataset(
            outdir,
            transform=None,
            pre_transform=None,
            pre_filter=None,
            datalist=datalist,
            idx=start,
        )

if __name__=="__main__":
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Get a labelled PyG dataset from HIPO files containing MC events, using REC::Traj info as the graph data.')

    # Add arguments
    parser.add_argument('--infiles', type=str, default=None, nargs='*',
                        help='Paths to input HIPO files')
    parser.add_argument('--outdir', type=str, default='',
                        help='Output dataset directory path')
    parser.add_argument('--step', type=int, default=1000,
                        help='Step size for looping HIPO files')
    parser.add_argument('--start', type=int, default=0,
                        help='Starting index for writing output files')
    parser.add_argument('--nolabels', action='store_true',
                        help='Skip labelling which reads MC::Lund bank and set all labels to 0.')

    # Parse
    args = parser.parse_args()

    # Run
    main(args.infiles,args.outdir,args.step,start=args.start,nolabels=args.nolabels)
