import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim.lr_scheduler import _LRScheduler
from pytorch_lightning import LightningModule, Trainer, seed_everything
import math
import time
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler, BatchSampler
from argoverse.utils.se2 import SE2
#from argoverse.evaluation.competition_util import generate_forecasting_h5

class GoalTransformer(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.goal_transformer = GoalTransformerBase(
            agent_input_dim=4,
            static_map_input_dim=40,
            output_dim=9,
            T=params['t_hist'],
            A=params['max_agents'],
            G=params['max_centerlines'],
            D=params['feature_dim'],
            H=params['nheads'],
            P=params['npred'],
            k=4,
            dt=0.1,
            num_goals=params['num_goals'] if not params['skip_gp_head'] else params['t_pred'],
            t_pred=params['t_pred'],
            dropout=params['dropout'],
            activation='relu',
            sigma_obs=params['sigma_obs'], # Fixed observation error std. Might want to learn that as well.
            skip_gp_head=False if not 'skip_gp_head' in params else params['skip_gp_head'],  # Whether to skip GP head or not
            cross_terms=params.get('cross_terms', True)
        )
        self.criterion = Criterion()
        self.metrics = Metrics()
        num_clusters = params['npred']
        cluster_centers = torch.load(f'data/clusters/kmeans_position_velocity_{num_clusters:03d}_clusters.pt')
        self.register_buffer('cluster_centers', torch.nn.Parameter(cluster_centers))

        self.params = params

    def forward(self, agents, static_map, agents_mask=None, static_map_mask=None):
        prediction = self.goal_transformer(agents, static_map, agents_mask=agents_mask, static_map_mask=static_map_mask)
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params['lr'], 
                                      weight_decay=self.params['weight_decay'])
        scheduler = InverseSquareRootLR(optimizer, self.params['lr_warmup_steps'])
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                    'reduce_on_plateau': False,
                    'monitor': 'total_loss_train',
                }
            ]
        )

    def training_step(self, train_batch, batch_idx):
        conf_loss_weight = self.params['conf_loss_weight']
        mse_loss = self.params['mse_loss']
        ground_truth = train_batch['ground_truth'] # This should be [BATCH_SIZE,t_pred,4] - (x,y,u,v)
        agents = train_batch['agents']
        static_map = train_batch['static_map']
        agents_mask = train_batch['agents_mask']
        static_map_mask = train_batch['static_map_mask']

        trajectories, trajectory_cov, goals, goal_cov, prob = self.goal_transformer(agents, static_map, agents_mask=agents_mask, static_map_mask=static_map_mask)

        # Backprop is done from the full trajectory that is closest to the center of the ground truth cluster + the probability of that prediction
        cluster_ids = torch.argmin(torch.linalg.norm(ground_truth.unsqueeze(-3).flatten(-2,-1) - self.cluster_centers.flatten(-2,-1), dim=-1), dim=1)

        # Compute losses
        traj_loss, conf_loss = self.criterion(trajectories, trajectory_cov, ground_truth, prob, cluster_ids=cluster_ids, mse_loss=mse_loss)

        total_loss = traj_loss + conf_loss_weight * conf_loss

        # Compute metrics on training dataset
        minFDE, minADE, miss_rate, brier_minFDE, minFGNLL, minAGNLL = self.metrics(trajectories, trajectory_cov, ground_truth, prob)

        # Log losses
        self.log('traj_loss_train', traj_loss, sync_dist=True)
        self.log('conf_loss_train', conf_loss, sync_dist=True)
        self.log('total_loss_train', total_loss, sync_dist=True)
        self.log('minADE_train', minADE, sync_dist=True)
        self.log('minFDE_train', minFDE, sync_dist=True)
        self.log('miss_rate_train', miss_rate, sync_dist=True)
        self.log('brier_minFDE_train', brier_minFDE, sync_dist=True)
        self.log('minFGNLL_train', minFGNLL, sync_dist=True)
        self.log('minAGNLL_train', minAGNLL, sync_dist=True)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        conf_loss_weight = self.params['conf_loss_weight']
        mse_loss = self.params['mse_loss']
        ground_truth = val_batch['ground_truth']
        agents = val_batch['agents']
        static_map = val_batch['static_map']
        agents_mask = val_batch['agents_mask']
        static_map_mask = val_batch['static_map_mask']

        trajectories, trajectory_cov, goals, goal_cov, prob = self.goal_transformer(agents, static_map, agents_mask=agents_mask, static_map_mask=static_map_mask)
        
        # Compute losses
        traj_loss, conf_loss = self.criterion(trajectories, trajectory_cov, ground_truth, prob, cluster_ids=None, mse_loss=mse_loss)

        total_loss = traj_loss + conf_loss_weight * conf_loss

        # Compute metrics on training dataset
        minFDE, minADE, miss_rate, brier_minFDE, minFGNLL, minAGNLL = self.metrics(trajectories, trajectory_cov, ground_truth, prob)

        # Log losses
        self.log('traj_loss_val', traj_loss, sync_dist=True)
        self.log('conf_loss_val', conf_loss, sync_dist=True)
        self.log('total_loss_val', total_loss, sync_dist=True)
        self.log('minADE_val', minADE, sync_dist=True)
        self.log('minFDE_val', minFDE, sync_dist=True)
        self.log('miss_rate_val', miss_rate, sync_dist=True)
        self.log('brier_minFDE_val', brier_minFDE, sync_dist=True)
        self.log('minFGNLL_val', minFGNLL, sync_dist=True)
        self.log('minAGNLL_val', minAGNLL, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        forecasted_trajectories = {}
        forecasted_probabilities = {}

        max_n_guesses = 6
        horizon = self.params['t_pred']
        agents = test_batch['agents']
        static_map = test_batch['static_map']
        agents_mask = test_batch['agents_mask']
        static_map_mask = test_batch['static_map_mask']
        seq_id = test_batch['seq_id']
        origin = test_batch['origin'].cpu().numpy()

        BATCH_SIZE = agents.shape[0] # There is a better way to get the batch size

        trajectories, trajectory_cov, goals, goal_cov, prob = self.goal_transformer(agents, static_map, agents_mask=agents_mask, static_map_mask=static_map_mask)
        
        for sid in range(BATCH_SIZE):
            sample_origin = SE2(origin[sid,:2,:2], origin[sid,:2,-1])
            sample_trajectories = trajectories[sid].cpu()
            sample_prob = torch.nn.Softmax(dim=-1)(prob[sid])
            top_predictions = torch.argsort(sample_prob, descending=True)[:max_n_guesses]

            # Transform predictions to global frame
            pred_trajectories = np.zeros((max_n_guesses, horizon, 3), dtype=np.float32)
            for j, k in enumerate(top_predictions):
                pred_trajectories[j, :, :2] = sample_origin.transform_point_cloud(sample_trajectories[k,:,:2])
                pred_trajectories[j, :, 2] = sample_prob[k].cpu().item() / torch.sum(sample_prob[top_predictions]).cpu().item()

            # Add predictions to eval dicts
            forecasted_trajectories[seq_id[sid].item()] = pred_trajectories[:,:,:2]
            forecasted_probabilities[seq_id[sid].item()] = pred_trajectories[:,0,-1]

        return {'forecasted_trajectories': forecasted_trajectories,
                'forecasted_probabilities': forecasted_probabilities}


    def test_epoch_end(self, outputs):
        forecasted_trajectories = {}
        forecasted_probabilities = {}

        for x in outputs:
            forecasted_trajectories.update(x['forecasted_trajectories'])
            forecasted_probabilities.update(x['forecasted_probabilities'])

        # Save predictions
        torch.save((forecasted_trajectories, forecasted_probabilities), self.params['submission_dir'] + self.params['model_name'] + '_test_predictions.pt')

#       # Save predictions in submission-ready format
#       generate_forecasting_h5(
#          forecasted_trajectories,
#          params['submission_dir'],
#          params['model_name'],
#          forecasted_probabilities
#       )

#       print(f"Submission data saved in {params['submission_dir'] + params['model_name'] + '.h5'}")

    def train_dataloader(self):
        train_data_list = []
        dataset_dir = Path(self.params['dataset_dir'])
        for i in range(self.params['num_train_datasets']):
            train_data_list.append(
                torch.load(dataset_dir / f"train/argoverse_{i:02d}_dataset.pt")
            )
        
        train_dataset = ConcatDataset(train_data_list)
        train_loader = DataLoader(
            train_dataset, batch_size=self.params['batch_size'], shuffle=True, drop_last=True, num_workers=self.params['num_workers'], pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        val_data_list = []
        dataset_dir = Path(self.params['dataset_dir'])
        for i in range(self.params['num_val_datasets']):
            val_data_list.append(
                torch.load(dataset_dir / f"val/argoverse_{i:02d}_dataset.pt")
            )
        
        val_dataset = ConcatDataset(val_data_list)
        val_loader = DataLoader(
            val_dataset, batch_size=self.params['batch_size'], shuffle=False, drop_last=False, num_workers=self.params['num_workers'], pin_memory=True
        )
        return val_loader

    def test_dataloader(self):
        test_data_list = []
        dataset_dir = Path(self.params['dataset_dir'])
        for i in range(self.params['num_test_datasets']):
            test_data_list.append(
                torch.load(dataset_dir / f"test/argoverse_{i:02d}_dataset.pt")
            )
        
        test_dataset = ConcatDataset(test_data_list)
        test_loader = DataLoader(
            test_dataset, batch_size=self.params['batch_size'], shuffle=False, drop_last=False, num_workers=self.params['num_workers'], pin_memory=True
        )
        return test_loader


class InverseSquareRootLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        if warmup_steps <= 0:
            raise ValueError('warmup_steps must be > 0')
        self._warmup_steps = warmup_steps
        self._lr_steps = [param_group['lr'] / warmup_steps for param_group in optimizer.param_groups]
        self._decay_factors = [
            param_group['lr'] * warmup_steps ** 0.5 for param_group in optimizer.param_groups
        ]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.last_epoch < self._warmup_steps:
            return [self.last_epoch * lr_step for lr_step in self._lr_steps]
        else:
            return [decay_factor * self.last_epoch ** -0.5 for decay_factor in self._decay_factors]


class GoalTransformerBase(nn.Module):
    def __init__(self, agent_input_dim=4, static_map_input_dim=40, output_dim=9, T=20, 
                 A=64, G=16, D=128, H=4, P=1, k=4, dt=0.1, num_goals=1, t_pred=30, 
                 dropout=0.1, activation="relu", sigma_obs=[0.5,0.5,0.5,0.5], skip_gp_head=False, cross_terms=True):
        super().__init__()

        self.agent_input_dim = agent_input_dim
        self.static_map_input_dim = static_map_input_dim
        self.num_goals = num_goals # Number of itermediate goals and uncertainties to be predicted
        self.T = T # Max number of observed time steps
        self.A = A # Max number of agents (including target agent)
        self.G = G # Max number of static map elements (lanes)
        self.D = D # Feature size
        self.H = H # Number of heads
        self.P = P # Max number of predictions
        self.dt = dt # Time steps
        self.skip_gp_head = skip_gp_head
        
        # Modules
        self.agent_encoder = AgentEncoder(agent_input_dim, D, activation=activation)
        self.static_map_encoder = StaticMapEncoder(static_map_input_dim, D, T, activation=activation)
        self.encoder = TransformerEncoder(D, H, k=k, dropout=dropout, activation=activation)
        self.decoder = TransformerDecoder(A, T, D, H, P, k=k, output_dim=output_dim, num_goals=num_goals, dropout=dropout, activation=activation)
        self.query_embed = nn.Embedding(P,D)
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=D, normalize=True)

        if not self.skip_gp_head:
#           self.gaussian_process_head = GaussianProcessHead(T=T, P=P, dt=dt, num_goals=num_goals, t_pred=t_pred,
#                                                            sigma_obs=sigma_obs, cross_terms=cross_terms)
            self.gaussian_process_head = GaussianProcessHeadFast(T=T, P=P, dt=dt, num_goals=num_goals, t_pred=t_pred,
                                                             sigma_obs=sigma_obs, cross_terms=cross_terms)

        self._initialize()

    def _initialize(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, agents, static_map, agents_mask=None, static_map_mask=None, debug=False):

        # Dimension checks
        assert agents.shape[1] == self.A
        assert agents.shape[2] == self.T
        assert agents.shape[3] == self.agent_input_dim
        assert static_map.shape[1] == self.G
        assert static_map.shape[2] == self.static_map_input_dim
        BATCH_SIZE = agents.shape[0]
        
        # Repeat static map mask in time dimension
        if static_map_mask is not None:
#           static_map_mask = static_map_mask.unsqueeze(-1).expand(-1,-1,self.T)
            static_map_mask = static_map_mask.unsqueeze(-1).repeat(1,1,self.T)
        
        # Encode the inputs
        agent_features = self.agent_encoder(agents)
        static_map_features = self.static_map_encoder(static_map)
        
        # Get position encodings (for agents and time)
        agent_pos = self.position_embedding(agent_features)
        static_map_pos = self.position_embedding(static_map_features)
        
        # Transformer encoder
        agent_memory = self.encoder(
            agent_features, 
            static_map_features, 
            agents_mask=agents_mask, 
            static_map_mask=static_map_mask,
            agent_pos=agent_pos,
            static_map_pos=static_map_pos
        )
        
        # Pass prediction queries through transformer decoder to get intermediate goals
#       goal_queries = self.query_embed.weight.unsqueeze(0).expand(BATCH_SIZE,-1,-1)
        goal_queries = self.query_embed.weight.unsqueeze(0).repeat(BATCH_SIZE,1,1)
        goal_dec = self.decoder(
            goal_queries, 
            agent_memory, 
            agents_mask=agents_mask, 
            agent_pos=agent_pos
        )
        
        # Separate means and covariances
        goals = goal_dec[:,:,:,:4] # Mean x, y, u, v
        prob = goal_dec[:,:,-1,-1] # trajectory prob
        goal_cov = get_covariance_from_vel(goal_dec, varmin=0.2, varmax=1.0)
        
        # Now compute the rest of the trajectory with the Gaussian Processes
        if self.skip_gp_head:
            trajectories, trajectory_cov = goals, goal_cov
        else:
            trajectories, trajectory_cov = self.gaussian_process_head(agents, goals, goal_cov)
        
        return trajectories, trajectory_cov, goals, goal_cov, prob


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conf_loss = nn.CrossEntropyLoss()
        self.traj_loss = GNLLLoss()
        self.traj_loss_mse = nn.HuberLoss(delta=4.)
        
    def forward(self, trajectories, trajectory_cov, ground_truth, prob, cluster_ids=None, mse_loss=False):

        BATCH_SIZE, P, t_pred, _ = trajectories.shape
        q = torch.argmax(prob, dim=1) if cluster_ids is None else cluster_ids
        closest_trajectories = trajectories[range(BATCH_SIZE), q]
        closest_trajectory_cov = trajectory_cov[range(BATCH_SIZE), q]

        if mse_loss:
            traj_loss = self.traj_loss_mse(closest_trajectories[:,:,:2], ground_truth[:,:,:2])
        else:
            traj_loss = self.traj_loss(closest_trajectories, closest_trajectory_cov, ground_truth)

        conf_loss = self.conf_loss(prob, q)
        
        return traj_loss, conf_loss


class GNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, trajectory, trajectory_cov, trajectory_ground_truth, reduction='mean'):
        BATCH_SIZE = trajectory.shape[0]
        T = trajectory.shape[1]

        loss = 0.0
        
        # Position loss
        eps = 2.e-1
#       I_eps = eps * torch.eye(2).expand(BATCH_SIZE,T,-1,-1).to(trajectory.device)
        I_eps = eps * torch.eye(2).to(trajectory.device)
        tril_xy = torch.linalg.cholesky(trajectory_cov[...,:2,:2] + I_eps)
#       tril_xy = torch.linalg.cholesky(trajectory_cov[...,:2,:2])
        dist_xy = torch.distributions.MultivariateNormal(trajectory[...,:2], scale_tril=tril_xy)
        loss -= dist_xy.log_prob(trajectory_ground_truth[...,:2])

        if reduction == 'mean':
            return torch.sum(loss)/BATCH_SIZE/T
        else:
            return loss


class Metrics(nn.Module):
    def __init__(self):
        super().__init__()
        self.gnll_loss = GNLLLoss()
        
    def forward(self, trajectories, trajectory_cov, target, probabilities):

        BATCH_SIZE = trajectories.shape[0]
        q, idx = torch.sort(probabilities, dim=-1, descending=True)

        FDE = torch.zeros(BATCH_SIZE,6)
        ADE = torch.zeros(BATCH_SIZE,6)
        brier_FDE = torch.zeros(BATCH_SIZE,6)
        FGNLL = torch.zeros(BATCH_SIZE,6)
        AGNLL = torch.zeros(BATCH_SIZE,6)

        probabilities_scaled = torch.nn.Softmax(dim=-1)(probabilities)

        for i in range(6):
            closest_traj = trajectories[range(BATCH_SIZE), idx[:,i],:,:2]
            closest_prob = probabilities_scaled[range(BATCH_SIZE), idx[:,i]]
            FDE[:,i] = torch.norm(closest_traj[:,-1] - target[:,-1,:2], 2, -1)
            ADE[:,i] = torch.mean(torch.norm(closest_traj - target[:,:,:2], 2, -1), dim=-1)
            brier_FDE[:,i] = torch.norm(closest_traj[:,-1] - target[:,-1,:2], 2, -1) + (1. - closest_prob).pow(2)
            FGNLL[:,i] = self.gnll_loss(trajectories[range(BATCH_SIZE), idx[:,i], -1, :2], trajectory_cov[range(BATCH_SIZE), idx[:,i], -1, :2, :2], target[:,-1,:2])
            AGNLL[:,i] = torch.mean(self.gnll_loss(trajectories[range(BATCH_SIZE), idx[:,i], :, :2], trajectory_cov[range(BATCH_SIZE), idx[:,i], :, :2, :2], target[:, :, :2]), dim=-1)

        minFDE = torch.mean(torch.min(FDE, dim=-1)[0])
        minADE = torch.mean(torch.min(ADE, dim=-1)[0])
        brier_minFDE = torch.mean(torch.min(brier_FDE, dim=-1)[0])
        minFGNLL = torch.mean(torch.min(FGNLL, dim=-1)[0])
        minAGNLL = torch.mean(torch.min(AGNLL, dim=-1)[0])
        miss_rate = torch.sum(torch.all(FDE > 2.0, dim=-1)) / BATCH_SIZE

        return minFDE, minADE, miss_rate, brier_minFDE, minFGNLL, minAGNLL
        

class AgentEncoder(nn.Module):
    def __init__(self, input_dim, D, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 4*D)
        self.batch_norm1 = BatchNorm2d(4*D)
        self.linear2 = nn.Linear(4*D, 4*D)
        self.batch_norm2 = BatchNorm2d(4*D)
        self.linear3 = nn.Linear(4*D, D)
        self.activation = _get_activation_fn(activation)
    
    def forward(self, x):
        """x: [BATCH_SIZE, A, T, INPUT_DIM]"""
        x = self.activation(self.batch_norm1(self.linear1(x)))
        return self.linear3(self.activation(self.batch_norm2(self.linear2(x))))
    
    
class StaticMapEncoder(nn.Module):
    def __init__(self, input_dim, D, T, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 4*D)
        self.batch_norm1 = BatchNorm1d(4*D)
        self.linear2 = nn.Linear(4*D, 4*D)
        self.batch_norm2 = BatchNorm1d(4*D)
        self.linear3 = nn.Linear(4*D, D)
        self.activation = _get_activation_fn(activation)
        self.T = T
    
    def forward(self, x):
        """x: [BATCH_SIZE, LANE_NUM, INPUT_DIM]"""
        encoding = self.activation(self.batch_norm1(self.linear1(x)))
        encoding = self.linear3(self.activation(self.batch_norm2(self.linear2(encoding))))

        # Tile across time steps and return
        return encoding.unsqueeze(-2).repeat(1, 1, self.T, 1)
    

class TransformerEncoder(nn.Module):
    """2D Attention is similar to this: https://arxiv.org/pdf/1912.12180.pdf"""
    def __init__(self, D, H, k=4, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Agent-Agent attention layers
        self.agent_attention = nn.ModuleList(
            [TransformerEncoderLayer2D(D, H, k=k, dropout=dropout, activation=activation) for i in range(4)]
        )
        
        # Agent-Map attention layers
        self.static_map_attention = nn.ModuleList(
            [TransformerEncoderLayerMap(D, H, k=k, dropout=dropout, activation=activation) for i in range(2)]
        )
        
        self.norm = nn.LayerNorm(D)
        
    def forward(self, agents, static_map, agents_mask=None, static_map_mask=None, agent_pos=None, static_map_pos=None):
        """Agents and static maps are already encoded"""
        BATCH_SIZE, A, T, D = agents.size()
        _, G, _, _ = static_map.size()
        
        # Agents
        for i in range(2):
            agents = self.agent_attention[i](agents, agents_mask=agents_mask, agent_pos=agent_pos)
        
        # Map
        agents = self.static_map_attention[0](agents, static_map, agents_mask=agents_mask, static_map_mask=static_map_mask, agent_pos=agent_pos, static_map_pos=static_map_pos)
        
        # Agents
        agents = self.agent_attention[2](agents, agents_mask=agents_mask, agent_pos=agent_pos)
        
        # Map
        agents = self.static_map_attention[1](agents, static_map, agents_mask=agents_mask, static_map_mask=static_map_mask, agent_pos=agent_pos, static_map_pos=static_map_pos)
        
        # Agents
        agents = self.agent_attention[3](agents, agents_mask=agents_mask, agent_pos=agent_pos)

        agents = self.norm(agents)
        
        return agents


class TransformerDecoder(nn.Module):
    def __init__(self, A, T, D, H, P, output_dim=9, num_goals=1, k=4, dropout=0.1, activation="relu"):
        super().__init__()
        
        self.P = P
        self.output_dim = output_dim
        self.num_goals = num_goals
        self.linear11 = nn.Linear(D, k*D)
        self.linear12 = nn.Linear(k*D, D)
        self.batch_norm1 = BatchNorm1d(k*D)
        self.linear2 = nn.Linear(D, k*D)
        self.linear3 = nn.Linear(k*D, output_dim*num_goals)
        self.batch_norm2 = BatchNorm1d(k*D)
        
        # Prediction self-attention and cross-attention
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(A, T, D, H, k=k, dropout=dropout, activation=activation) for i in range(3)]
        )
                
        self.activation = _get_activation_fn(activation)
    
        
    def forward(self, goal_queries, agents, agents_mask=None, agent_pos=None):
        BATCH_SIZE, A, T, D = agents.size()
        
#         pred_queries = self.norm1(pred_queries)
        goal_queries = self.linear12(self.activation(self.batch_norm1(self.linear11(goal_queries))))
    
        for decoder_layer in self.decoder_layers:
            goal_queries = decoder_layer(goal_queries, agents, agents_mask=agents_mask, agent_pos=agent_pos)
            
        goals = self.linear3(self.activation(self.batch_norm2(self.linear2(goal_queries))))
        goals = goals.reshape(BATCH_SIZE,-1,self.num_goals,self.output_dim)
        
        return goals

    
class TransformerEncoderLayer2D(nn.Module):
    def __init__(self, D, H, k=4, dropout=0, activation="relu"):
        super().__init__()
        self.D = D
        self.H = H
        
        self.attention_across_time = SelfAttention(D, H, k=k, dropout=dropout, activation=activation)
        self.attention_across_agents = SelfAttention(D, H, k=k, dropout=dropout, activation=activation)
    
    def forward(self, agents, agents_mask=None, agent_pos=None):
        BATCH_SIZE, A, T, D = agents.shape
        
        # Attention across time
        x = agents.reshape(-1,T,D)
        pos = agent_pos.reshape(-1,T,D)
        key_mask = agents_mask.reshape(-1,T)
        x = self.attention_across_time(x, key_mask=key_mask, pos=pos) # This has dimension [B*A, T, D]
        agents = x.reshape(BATCH_SIZE, A, T, D)
        
        # Attention across agents
        x = agents.permute(0,2,1,3).contiguous().reshape(-1,A,D)
        pos = agent_pos.permute(0,2,1,3).contiguous().reshape(-1,A,D)
        key_mask = agents_mask.permute(0,2,1).contiguous().reshape(-1,A)
        x = self.attention_across_agents(x, key_mask=key_mask, pos=pos) # This has dimension [B*T, A, D]
        agents = x.reshape(BATCH_SIZE, T, A, D).permute(0,2,1,3).contiguous()
        
        return agents
    

class TransformerEncoderLayerMap(nn.Module):
    def __init__(self, D, H, k=4, dropout=0, activation="relu"):
        super().__init__()
        self.D = D
        self.H = H
        self.cross_attention = CrossAttention(D, H, k=k, dropout=dropout, activation=activation)
    
    def forward(self, agents, static_map, agents_mask=None, static_map_mask=None, agent_pos=None, static_map_pos=None):
        BATCH_SIZE, A, T, D = agents.shape
        G = static_map.shape[1]
        
        x = agents.permute(0,2,1,3).contiguous().reshape(-1,A,D)
        query_pos = agent_pos.permute(0,2,1,3).contiguous().reshape(-1,A,D)
        
        # Assuming that a map is available at every time step
        kv = static_map.permute(0,2,1,3).contiguous().reshape(-1,G,D)
        key_pos = static_map_pos.permute(0,2,1,3).contiguous().reshape(-1,G,D)
        key_mask = static_map_mask.permute(0,2,1).contiguous().reshape(-1,G)
        
        x = self.cross_attention(x, kv, key_mask=key_mask, query_pos=query_pos, key_pos=key_pos)
        agents = x.reshape(BATCH_SIZE, T, A, D).permute(0,2,1,3).contiguous()
        
        return agents


class TransformerDecoderLayer(nn.Module):
    def __init__(self, A, T, D, H, k=4, dropout=0, activation="relu"):
        super().__init__()
        self.self_attention = SelfAttention(D, H, k=k, dropout=dropout, activation=activation)
        self.cross_attention = CrossAttention(D, H, k=k, dropout=dropout, activation=activation)
        
        self.A, self.T, self.D, self.H = A, T, D, H
    
    def forward(self, pred_queries, agents, pred_queries_mask=None, agents_mask=None, pred_queries_pos=None, agent_pos=None):
        
        # Self-attention for pred_queries (no reshaping needed)
        q = pred_queries
        y = self.self_attention(q, key_mask=pred_queries_mask, pos=pred_queries_pos)
        
        # Cross-attention from pred_queries to agent memory
        q = pred_queries
        kv = agents.reshape(-1,self.A * self.T, self.D)
        key_pos = agent_pos.reshape(-1,self.A * self.T, self.D)
        key_mask = agents_mask.reshape(-1,self.A * self.T)
        pred_queries = self.cross_attention(q, kv, key_mask=key_mask, query_pos=pred_queries_pos, key_pos=key_pos)
        
        return pred_queries
    
    
class SelfAttention(nn.Module):
    def __init__(self, D, H, k=4, dropout=0, activation="relu"):
        super().__init__()
        self.D = D
        self.H = H
        self.multihead_attention = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.linear = nn.ModuleList([nn.Linear(D, k*D), nn.Linear(k*D, D)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
        self.norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(2)])
        
        self.activation = _get_activation_fn(activation)
        
    def _add_pos_embeddings(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, key_mask=None, pos=None):
        x0 = self.norm[0](x)
        to_attend = torch.ones(x.shape[0], dtype=torch.bool) if key_mask is None else ~torch.all(key_mask, dim=-1)
        q = k = self._add_pos_embeddings(x0, pos)[to_attend]
        v = x0[to_attend]
        key_mask = None if key_mask is None else key_mask[to_attend] 
        y = torch.zeros_like(x)
        y[to_attend] = self.multihead_attention(q, k, value=v, key_padding_mask=key_mask)[0]
        x = x + self.dropout[0](y)
        z = self.linear[1](self.dropout[1](self.activation(self.linear[0](self.norm[1](x)))))
        x = x + self.dropout[2](z)
        
        return x

class CrossAttention(nn.Module):
    def __init__(self, D, H, k=4, dropout=0, activation="relu"):
        super().__init__()
        self.D = D
        self.H = H
        self.multihead_attention = nn.MultiheadAttention(D, H, dropout=dropout, batch_first=True)
        self.linear = nn.ModuleList([nn.Linear(D, k*D), nn.Linear(k*D, D)])
        self.dropout = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
        self.norm = nn.ModuleList([nn.LayerNorm(D) for _ in range(3)])
        
        self.activation = _get_activation_fn(activation)
        
    def _add_pos_embeddings(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, kv, key_mask=None, query_pos=None, key_pos=None):
        x0 = self.norm[0](x)
        kv0 = self.norm[1](kv)
        q = self._add_pos_embeddings(x0, query_pos) # Shape: [B*T, A, D]
        k = self._add_pos_embeddings(kv0, key_pos) # Shape: [B*T, G, D]
        v = kv0 # Shape: same as k

        y = self.multihead_attention(q, k, value=v, key_padding_mask=key_mask)[0] # Shape: [B*T, A, D]
        x = x + self.dropout[0](y)
        z = self.linear[1](self.dropout[1](self.activation(self.linear[0](self.norm[1](x)))))
        x = x + self.dropout[2](z)
        
        return x
    

class GaussianProcessHeadFast(nn.Module):
    def __init__(self, T=20, P=1, dt=0.1, num_goals=1, t_pred=30,
                 sigma_obs=[0.5,0.5,0.5,0.5], cross_terms=True):
        super().__init__()
        
        self.P = P # Number of predictions
        self.T = T # Number of observed time steps
        self.dt = dt # Time step
        self.num_goals = num_goals # Number of intermediate goals
        self.t_pred = t_pred # Number of time steps to predict
        self.sigma_obs = sigma_obs # Std of observations (fixed for now)
        self.gaussian_process_interpolator = GaussianProcessBase(T, t_pred, dt=dt, num_goals=num_goals, cross_terms=cross_terms)
        
        # Observation cov (fixed for now)
        self.register_buffer('cov_obs', torch.diag(torch.tensor(self.sigma_obs).repeat(self.T)).unsqueeze(0)**2)
        
    def forward(self, agents, goals, goal_cov):
        device = agents.device
        BATCH_SIZE = agents.shape[-4]
        
        # Prepare observations: same for all Gaussian processes
        y_obs = agents[:,0].unsqueeze(1).repeat(1,self.P,1,1).view(-1,self.T,4) # Keep the first agent only - the target agent
        cov_obs = self.cov_obs.repeat(BATCH_SIZE*self.P,1,1)
        
        # Prepare goals
        y_goal = goals[:,:,:,:4].view(-1,self.num_goals,4)

        # The following is much faster for batch block diagonal of batches
        cov_goal = (
                torch.diag_embed(torch.diagonal(goal_cov, dim1=-2, dim2=-1).flatten(-2,-1))
              + torch.diag_embed(
                    torch.nn.functional.pad(torch.diagonal(goal_cov, offset=-1, dim1=-2, dim2=-1), (0,1), "constant", 0).flatten(-2,-1),
                    offset=-1
                )[:,:,:-1,:-1]
              + torch.diag_embed(
                    torch.nn.functional.pad(torch.diagonal(goal_cov, offset=-2, dim1=-2, dim2=-1), (0,2), "constant", 0).flatten(-2,-1),
                    offset=-2
                )[:,:,:-2,:-2]
              + torch.diag_embed(
                    torch.nn.functional.pad(torch.diagonal(goal_cov, offset=-3, dim1=-2, dim2=-1), (0,3), "constant", 0).flatten(-2,-1),
                    offset=-3
                )[:,:,:-3,:-3]
              + torch.diag_embed(
                    torch.nn.functional.pad(torch.diagonal(goal_cov, offset=1, dim1=-2, dim2=-1), (0,1), "constant", 0).flatten(-2,-1),
                    offset=1
                )[:,:,:-1,:-1]
              + torch.diag_embed(
                    torch.nn.functional.pad(torch.diagonal(goal_cov, offset=2, dim1=-2, dim2=-1), (0,2), "constant", 0).flatten(-2,-1),
                    offset=2
                )[:,:,:-2,:-2]
              + torch.diag_embed(
                    torch.nn.functional.pad(torch.diagonal(goal_cov, offset=3, dim1=-2, dim2=-1), (0,3), "constant", 0).flatten(-2,-1),
                    offset=3
                )[:,:,:-3,:-3]
        ).flatten(0,1)

        mean, cov = self.gaussian_process_interpolator(y_obs, cov_obs, y_goal, cov_goal)

        trajectories = mean.view(BATCH_SIZE,self.P,self.t_pred,4)

        # This is much faster
        trajectory_cov = (
          + torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1).view(-1,30,4))    
          + torch.diag_embed(
                torch.nn.functional.pad(torch.diagonal(cov, offset=-1, dim1=-2, dim2=-1), (0,1), "constant", 0).view(-1,30,4)[:,:,:3],
                offset=-1
            )
          + torch.diag_embed(
                torch.nn.functional.pad(torch.diagonal(cov, offset=-2, dim1=-2, dim2=-1), (0,2), "constant", 0).view(-1,30,4)[:,:,:2],
                offset=-2
            )
          + torch.diag_embed(
                torch.nn.functional.pad(torch.diagonal(cov, offset=-3, dim1=-2, dim2=-1), (0,3), "constant", 0).view(-1,30,4)[:,:,:1],
                offset=-3
            )
          + torch.diag_embed(
                torch.nn.functional.pad(torch.diagonal(cov, offset=1, dim1=-2, dim2=-1), (0,1), "constant", 0).view(-1,30,4)[:,:,:3],
                offset=1
            )
          + torch.diag_embed(
                torch.nn.functional.pad(torch.diagonal(cov, offset=2, dim1=-2, dim2=-1), (0,2), "constant", 0).view(-1,30,4)[:,:,:2],
                offset=2
            )
          + torch.diag_embed(
                torch.nn.functional.pad(torch.diagonal(cov, offset=3, dim1=-2, dim2=-1), (0,3), "constant", 0).view(-1,30,4)[:,:,:1],
                offset=3
            )
        ).view(BATCH_SIZE,self.P,self.t_pred,4,4)

        return trajectories, trajectory_cov


class GaussianProcessHead(nn.Module):
    def __init__(self, T=20, P=1, dt=0.1, num_goals=1, t_pred=30,
                 sigma_obs=[0.5,0.5,0.5,0.5], cross_terms=True):
        super().__init__()
        
        self.P = P # Number of predictions
        self.T = T # Number of observed time steps
        self.dt = dt # Time step
        self.num_goals = num_goals # Number of intermediate goals
        self.t_pred = t_pred # Number of time steps to predict
        self.sigma_obs = sigma_obs # Std of observations (fixed for now)
        self.gaussian_process_list = nn.ModuleList([GaussianProcessBase(T, t_pred, dt=dt, num_goals=num_goals, cross_terms=cross_terms) for i in range(P)])
        
        # Observation cov (fixed for now)
        self.register_buffer('cov_obs', torch.diag(torch.tensor(self.sigma_obs).repeat(self.T)).unsqueeze(0)**2)
        
    def forward(self, agents, goals, goal_cov):
        BATCH_SIZE = agents.shape[-4]
        
        # Prepare observations: same for all Gaussian processes
        y_obs = agents[:,0] # Keep the first agent only - the target agent
        cov_obs = self.cov_obs.repeat(BATCH_SIZE,1,1)
        
        # Prepare goals
        trajectories = []
        trajectory_cov = []
        
        for p in range(self.P):
            y_goal = goals[:,p,:,:4]
            cov_goal = (
                    torch.diag_embed(torch.diagonal(goal_cov[:,p], dim1=-2, dim2=-1).flatten(-2,-1))
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(goal_cov[:,p], offset=-1, dim1=-2, dim2=-1), (0,1), "constant", 0).flatten(-2,-1),
                        offset=-1
                    )[:,:-1,:-1]
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(goal_cov[:,p], offset=-2, dim1=-2, dim2=-1), (0,2), "constant", 0).flatten(-2,-1),
                        offset=-2
                    )[:,:-2,:-2]
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(goal_cov[:,p], offset=-3, dim1=-2, dim2=-1), (0,3), "constant", 0).flatten(-2,-1),
                        offset=-3
                    )[:,:-3,:-3]
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(goal_cov[:,p], offset=1, dim1=-2, dim2=-1), (0,1), "constant", 0).flatten(-2,-1),
                        offset=1
                    )[:,:-1,:-1]
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(goal_cov[:,p], offset=2, dim1=-2, dim2=-1), (0,2), "constant", 0).flatten(-2,-1),
                        offset=2
                    )[:,:-2,:-2]
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(goal_cov[:,p], offset=3, dim1=-2, dim2=-1), (0,3), "constant", 0).flatten(-2,-1),
                        offset=3
                    )[:,:-3,:-3]
            )
            
            mean, cov = self.gaussian_process_list[p](y_obs, cov_obs, y_goal, cov_goal)

            trajectories.append(mean.clone())
            trajectory_cov.append((
                    torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1).view(-1,30,4))    
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(cov, offset=-1, dim1=-2, dim2=-1), (0,1), "constant", 0).view(-1,30,4)[:,:,:3],
                        offset=-1
                    )
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(cov, offset=-2, dim1=-2, dim2=-1), (0,2), "constant", 0).view(-1,30,4)[:,:,:2],
                        offset=-2
                    )
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(cov, offset=-3, dim1=-2, dim2=-1), (0,3), "constant", 0).view(-1,30,4)[:,:,:1],
                        offset=-3
                    )
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(cov, offset=1, dim1=-2, dim2=-1), (0,1), "constant", 0).view(-1,30,4)[:,:,:3],
                        offset=1
                    )
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(cov, offset=2, dim1=-2, dim2=-1), (0,2), "constant", 0).view(-1,30,4)[:,:,:2],
                        offset=2
                    )
                  + torch.diag_embed(
                        torch.nn.functional.pad(torch.diagonal(cov, offset=3, dim1=-2, dim2=-1), (0,3), "constant", 0).view(-1,30,4)[:,:,:1],
                        offset=3
                    )
                ).view(BATCH_SIZE,self.t_pred,4,4)
                 
            )

        trajectories = torch.stack(trajectories, dim=1)
        trajectory_cov = torch.stack(trajectory_cov, dim=1)
        
        return trajectories, trajectory_cov

    
class GaussianProcessBase(nn.Module):
    def __init__(self, t_hist, t_pred, dt=0.1, num_goals=1, cross_terms=True):
        super().__init__()

        # Use dynamics-informed kernel?
        self.cross_terms = cross_terms
        
        # Trainable hyperparameters
        self.tau = nn.Parameter(torch.tensor([1.5]), requires_grad=False)
        self.theta = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        self.theta_dot = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=True)
        
        # Time vectors
        t_obs = torch.linspace(0, dt * (t_hist - 1), t_hist)
        t_goal = torch.tensor([(t_hist-1)*dt + (i+1) * (t_pred//num_goals) * dt for i in range(num_goals)])
        self.register_buffer('t_train', torch.cat((t_obs, t_goal), dim=-1))
        self.register_buffer('t', torch.arange(t_hist * dt,(t_hist + t_pred) * dt, dt))

        NT = max(self.t_train.shape[-1], self.t.shape[-1])
        self.register_buffer('Ixx', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[1, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Ixu', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 1, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Iux', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [1, 0, 0, 0], 
                                                            [0, 0, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Iuu', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 1, 0], 
                                                            [0, 0, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Iyy', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 0, 0], 
                                                            [0, 1, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Iyv', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 0, 0], 
                                                            [0, 0, 0, 1], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Ivy', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 1, 0, 0], ])).unsqueeze(0)
        )
        self.register_buffer('Ivv', 
              torch.kron(torch.ones((NT,NT)), torch.tensor([[0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 0], 
                                                            [0, 0, 0, 1], ])).unsqueeze(0)
        )
        
    def forward(self, y_obs, cov_obs, y_goal, cov_goal):
        BATCH_SIZE = y_obs.shape[0]
        device = y_obs.device
        
        # Prepare training data
        y_train = torch.cat((y_obs, y_goal), dim=-2).flatten(1,2)
        var_train = torch.stack([torch.block_diag(co, cg) for co, cg in zip(cov_obs, cov_goal)])

        # Compute Kernels
        K_tt = self.kernel(self.t, self.t).expand(BATCH_SIZE,-1,-1)
        K_tT = self.kernel(self.t, self.t_train).expand(BATCH_SIZE,-1,-1)
        K_TT = self.kernel(self.t_train, self.t_train).expand(BATCH_SIZE,-1,-1)
        K_TT_inv = torch.inverse(K_TT + var_train)

        mu = (K_tT.bmm(K_TT_inv).bmm(y_train.unsqueeze(-1)).squeeze(-1)).reshape(BATCH_SIZE,-1,4)
        cov = K_tt - K_tT.bmm(K_TT_inv).bmm(K_tT.transpose(-1,-2))

        return mu, cov

    def kernel(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        device = t1.device

        N1 = t1.shape[-1]
        N2 = t2.shape[-1]
        t1 = t1.unsqueeze(-1).repeat(1,1,4).reshape(-1,1)
        t2 = t2.unsqueeze(-1).repeat(1,1,4).reshape(1,-1)
        t1til = t1 + self.tau
        t2til = t2 + self.tau

        Ixx = self.Ixx[:,:4*N1,:4*N2]
        Ixu = self.Ixu[:,:4*N1,:4*N2]
        Iux = self.Iux[:,:4*N1,:4*N2]
        Iuu = self.Iuu[:,:4*N1,:4*N2]
        Iyy = self.Iyy[:,:4*N1,:4*N2]
        Iyv = self.Iyv[:,:4*N1,:4*N2]
        Ivy = self.Ivy[:,:4*N1,:4*N2]
        Ivv = self.Ivv[:,:4*N1,:4*N2]
        
        theta_x = self.theta[0].clone()
        theta_dot_x = self.theta_dot[0].clone()
        theta_y = self.theta[1].clone()
        theta_dot_y = self.theta_dot[1].clone()
        
        kxx = theta_x ** 2 * (
            torch.minimum(t1til, t2til)**3 / 3. + torch.abs(t1 - t2) * torch.minimum(t1til, t2til)**2 / 2.
        )
        kxu = theta_x * theta_dot_x * (
            (t1 < t2).float() * t1**2 / 2. + (t1 >= t2).float() * (t1.mm(t2) - t2**2 / 2.)
        )
        kux = theta_x * theta_dot_x * (
            (t2 < t1).float() * t2**2 / 2. + (t2 >= t1).float() * (t1.mm(t2) - t1**2 / 2.)
        )
        kuu = theta_dot_x**2 * torch.minimum(t1, t2)
        
        kyy = theta_y ** 2 * (
            torch.minimum(t1til, t2til)**3 / 3. + torch.abs(t1 - t2) * torch.minimum(t1til, t2til)**2 / 2.
        )
        kyv = theta_y * theta_dot_y * (
            (t1 < t2).float() * t1**2 / 2. + (t1 >= t2).float() * (t1.mm(t2) - t2**2 / 2.)
        )
        kvy = theta_y * theta_dot_y * (
            (t2 < t1).float() * t2**2 / 2. + (t2 >= t1).float() * (t1.mm(t2) - t1**2 / 2.)
        )
        kvv = theta_dot_y**2 * torch.minimum(t1, t2)

        if self.cross_terms:
            K = Ixx.mul(kxx) + Ixu.mul(kxu) + Iux.mul(kux) + Iuu.mul(kuu) + Iyy.mul(kyy) + Iyv.mul(kyv) + Ivy.mul(kvy) + Ivv.mul(kvv)
        else:
            K = Ixx.mul(kxx) + Iuu.mul(kxx) + Iyy.mul(kyy) + Ivv.mul(kyy)

        return K


class BatchNorm1d(nn.Module):
    """
    1D BatchNorm, but with feature dimension being last
    """
    def __init__(self, D):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(D)
    
    def forward(self, x):
        return self.batch_norm(x.permute(0,2,1)).permute(0,2,1)
    

class BatchNorm2d(nn.Module):
    """
    2D BatchNorm, but with feature dimension being last
    """
    def __init__(self, D):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(D)
    
    def forward(self, x):
        return self.batch_norm(x.permute(0,3,1,2)).permute(0,2,3,1)
    

class PositionEmbeddingSine(nn.Module):
    """
    Standard positional embedding.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):

        a_embed = torch.ones_like(x).cumsum(1, dtype=torch.float32)
        t_embed = torch.ones_like(x).cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            a_embed = a_embed / (a_embed[:, -1:, :] + eps) * self.scale
            t_embed = t_embed / (t_embed[:, :, -1:] + eps) * self.scale

        dim = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim = torch.div(dim, 2, rounding_mode='floor')
        dim = self.temperature ** (2 * dim / self.num_pos_feats)

        pos_t = t_embed[:, :, :] / dim
        pos_a = a_embed[:, :, :] / dim
        pos_t = torch.stack((pos_t[:, :, :, 0::2].sin(), pos_t[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_a = torch.stack((pos_a[:, :, :, 0::2].sin(), pos_a[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return pos_a + pos_t
    
    
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def rot(theta):
    return torch.tensor([
        [torch.cos(theta), -torch.sin(theta)],
        [torch.sin(theta), torch.cos(theta)]
    ])
    
def get_covariance(goal_dec):
    '''
    input: goal_dec (BATCH_SIZE,P,t_pred,11)
    output: cov (BATCH_SIZE,P,t_pred,4,4)
    Note: covariances based on correlation factor rho = cov(x,y)/var(x)/var(y)
    '''
    var = torch.exp(goal_dec[:,:,:,4:8])
    cov_xy = torch.tanh(goal_dec[:,:,:,8]) * torch.sqrt(var[:,:,:,0] * var[:,:,:,1])
    cov_uv = torch.tanh(goal_dec[:,:,:,9]) * torch.sqrt(var[:,:,:,2] * var[:,:,:,3])
    cov = torch.diag_embed(var)
    cov[:,:,:,0,1] = cov_xy
    cov[:,:,:,1,0] = cov_xy
    cov[:,:,:,2,3] = cov_uv
    cov[:,:,:,3,2] = cov_uv
    
    return cov
    
def get_covariance_from_vel(goal_dec, varmin=0.2, varmax=3):
    '''
    input: goal_dec (BATCH_SIZE,P,t_pred,9)
    output: cov (BATCH_SIZE,P,t_pred,4,4)
    Note: covariances aligned with velocity vectors

    Note 2: This if inefficient for autograd. Need something better.
    '''

    theta = torch.atan2(goal_dec[:,:,:,3], goal_dec[:,:,:,2])
    var = torch.exp(goal_dec[:,:,:,4:8]).clamp(varmin, varmax) # Add epsilon to variance for stability
    cov = torch.diag_embed(var)
    Rot = torch.stack((
            torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1),
            torch.stack((-torch.sin(theta), torch.cos(theta)), dim=-1)
        ), dim=-1)
    cov[:,:,:,:2,:2] = Rot.matmul(cov[:,:,:,:2,:2].clone()).matmul(Rot.transpose(-2,-1))
    cov[:,:,:,2:,2:] = Rot.matmul(cov[:,:,:,2:,2:].clone()).matmul(Rot.transpose(-2,-1))
    
    return cov
    
    
