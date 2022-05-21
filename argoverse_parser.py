import os
import torch
import numpy as np
import pickle
import time
import sys
from tqdm import tqdm
from mpi4py import MPI
import argparse
import yaml

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.se2 import SE2

from core.datasets import ArgoverseForecastingSequences
from core.parse_tools import extract_argoverse_sequence_augmented

# MPI Stuff
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="parse training data", action="store_true")
    parser.add_argument("--val", help="parse validation data", action="store_true")
    parser.add_argument("--test", help="parse test data", action="store_true")
    args = parser.parse_args()

    # Data parser configuration file
    with open('config/argoverse_data_parser.yaml', 'r') as f:
        parser_config = yaml.safe_load(f)

    if args.train:
        data_dir = os.path.join(parser_config['raw_dataset_dir'],'train/data')
        output_dir = os.path.join(parser_config['parsed_dataset_dir'],'train')
    elif args.val:
        data_dir = os.path.join(parser_config['raw_dataset_dir'],'val/data')
        output_dir = os.path.join(parser_config['parsed_dataset_dir'],'val')
    elif args.test:
        data_dir = os.path.join(parser_config['raw_dataset_dir'],'test_obs/data')
        output_dir = os.path.join(parser_config['parsed_dataset_dir'],'test')
    else:
        raise ValueError('Choose either --train, --val, or --test dataset to parse')

    loader = ArgoverseForecastingLoader(data_dir)

    if rank == 0:
        print('Total number of sequences: ', len(loader))

    hd_map = ArgoverseMap()
    dt = 0.1

    # Dataset
    dataset = ArgoverseForecastingSequences(
        max_agents=parser_config['max_agents'], 
        max_centerlines=parser_config['max_centerlines']
    )

    # Iterate over all data
    N = len(loader)
    dN = int(N / size)

    # Augment data?
    if args.train:
        augment = parser_config['augment']
    else:
        augment = False
    
    progress_bar = tqdm(range(rank*dN,(rank+1)*dN)) if rank < size - 1 else tqdm(range(rank*dN,N))
    print(f'Running data loader on processor {rank+1} / {size} from sequence {rank*dN} to {(rank+1)*dN}')
    for i in progress_bar:
        seq = loader[i]
        
        parsed_data_list = extract_argoverse_sequence_augmented(seq, parser_config, hd_map, 
                train=args.train, val=args.val, test=args.test, augment=augment)

        # Append data to dataset
        for parsed_data in parsed_data_list:
            dataset.append(parsed_data)
        
    # Save dataset
    torch.save(dataset, os.path.join(output_dir, f'argoverse_{rank:02d}_dataset.pt'))

