import os
import torch
from kmeans_pytorch import kmeans
import argparse
import yaml
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

def train_dataloader(params):
    train_data_list = []
    dataset_dir = Path(params['dataset_dir'])
    for i in range(params['num_train_datasets']):
        print('Loading dataset ', i)
        train_data_list.append(
            torch.load(dataset_dir / f"train/argoverse_{i:02d}_dataset.pt")
        )
    
    train_dataset = ConcatDataset(train_data_list)
    train_loader = DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True, drop_last=True, num_workers=params['num_workers']
    )
    return train_loader

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/goal_transformer.yaml')
    parser.add_argument('--tol', type=float, default=0.001)
    args = parser.parse_args()
    
    # Load config file
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            params = yaml.safe_load(f)   
    else:
        raise ValueError('Configuration file does not exist.')
    
    # Number of clusters to generate
    num_clusters = params['npred']
    
    # Load training dataset
    BATCH_SIZE = params['batch_size']
    train_loader = train_dataloader(params)
    num_seq = len(train_loader) * BATCH_SIZE
    
    # Collect "future trajectories" in one tensor
    ground_truth = torch.zeros(num_seq, params['t_pred'] * 4)
    
    for i, data in enumerate(train_loader):
        ground_truth[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = data['ground_truth'].flatten(-2,-1)
    
    # Cluster and save
#   _, cluster_centers = kmeans(X=ground_truth, num_clusters=num_clusters, distance='euclidean', tol=args.tol, device=torch.device('cuda:0'))
    _, cluster_centers = kmeans(X=ground_truth, num_clusters=num_clusters, distance='euclidean', tol=args.tol)
    cluster_centers = cluster_centers.reshape(num_clusters,-1,4)
    torch.save(cluster_centers, f'data/clusters/kmeans_position_velocity_{num_clusters:03d}_clusters.pt')
    
        
