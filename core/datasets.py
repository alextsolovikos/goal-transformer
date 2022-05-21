import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class ArgoverseForecastingSequences(Dataset):
    """
    Torch dataset for pedestrian motion
    NEW VERSION: Set masks etc a priori (when appending data) to avoid excess computation
    during data loading.
    """
    def __init__(self, max_agents=32, max_centerlines=128, transform=None):
        
        # Vectorized data. Invalid places are NaN
        self.ground_truth = []
        self.ground_truth_raw = []
        self.agents = []
        self.static_map = []
        self.origin = []
        self.agents_mask = []
        self.static_map_mask = []
        self.city = []
        self.seq_id = []
        self.num_sequences = 0
        self.max_agents = max_agents
        self.max_centerlines = max_centerlines
        
        torch.set_default_dtype(torch.float32)

        
    def append(self, parsed_data):
        ground_truth = parsed_data['ground_truth']
        ground_truth_raw = parsed_data['ground_truth_raw']
        agents = parsed_data['agents']
        static_map = parsed_data['static_map']
        origin = parsed_data['origin']
        city = parsed_data['city']
        seq_id = parsed_data['seq_id']
        
        # Convert to torch
        if not torch.is_tensor(agents):
            agents = torch.from_numpy(agents)
        if not torch.is_tensor(static_map):
            static_map = torch.from_numpy(static_map)
        if not torch.is_tensor(origin):
            origin = torch.from_numpy(origin)
        if ground_truth is not None:
            if not torch.is_tensor(ground_truth):
                ground_truth = torch.tensor(ground_truth)
        if ground_truth_raw is not None:
            if not torch.is_tensor(ground_truth_raw):
                ground_truth_raw = torch.tensor(ground_truth_raw)

        # Order static_map centerlines by distance
        #     Note: static_map is still of dimension (128,10,4)
        min_distances = torch.min(torch.linalg.norm(static_map[:,:,:2], dim=-1), -1)[0]
        closest_lane_idx = torch.sort(min_distances)[1][:self.max_centerlines]
        static_map = static_map[closest_lane_idx]

        # Order agents by distance (first agent is always target agent)
        min_distances = torch.min(torch.linalg.norm(agents[1:,:,:2], dim=-1), -1)[0]
        closest_agent_idx = torch.sort(min_distances)[1][:self.max_agents-1] + 1
        closest_agent_idx = torch.cat((torch.tensor([0], dtype=int), closest_agent_idx))
        agents = agents[closest_agent_idx]

        # Pad or truncate if there are too many agents and map centerlines
        num_agents = agents.shape[0]
        num_centerlines = static_map.shape[0]
        agents = F.pad(agents, (0, 0, 0, 0, 0, self.max_agents - num_agents))
        static_map = F.pad(static_map, (0, 0, 0, 0, 0, self.max_centerlines - num_centerlines)).flatten(-2)
        
        # Masks
        agents_mask = torch.zeros((self.max_agents, 20), dtype=torch.bool)
        if num_agents < self.max_agents:
            agents_mask[num_agents:] = True
        static_map_mask = torch.zeros((self.max_centerlines), dtype=torch.bool)
        if num_centerlines < self.max_centerlines:
            static_map_mask[num_centerlines:] = True

        if static_map_mask.all():
            static_map_mask[0] = False # If no centerlines are available, add fake centerline to avoid nans

        # Append to data lists
        self.ground_truth.append(ground_truth if ground_truth is not None else -1)
        self.ground_truth_raw.append(ground_truth_raw if ground_truth_raw is not None else -1)
        self.agents.append(agents.type(torch.float32))
        self.static_map.append(static_map)
        self.origin.append(origin)
        self.agents_mask.append(agents_mask)
        self.static_map_mask.append(static_map_mask)
        self.city.append(city)
        self.seq_id.append(seq_id if seq_id is not None else -1)
        self.num_sequences += 1
        
    def set_max_limits(self, max_agents=32, max_centerlines=128):
#       self.max_agents = max_agents
#       self.max_centerlines = max_centerlines
        raise ValueError('Cannot change max agent limits after parsing')
        
    def __len__(self):
        return self.num_sequences
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
            'ground_truth': self.ground_truth[idx],
            'ground_truth_raw': self.ground_truth_raw[idx],
            'agents': self.agents[idx],
            'static_map': self.static_map[idx],
            'origin': self.origin[idx],
            'agents_mask': self.agents_mask[idx],
            'static_map_mask': self.static_map_mask[idx],
            'seq_id': self.seq_id[idx]
        }
        
        return sample
    
