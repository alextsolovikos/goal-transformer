import torch
import numpy as np
import pickle
import time
import sys
from tqdm import tqdm
import argparse
import yaml

from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.se2 import SE2

from core.datasets import ArgoverseForecastingSequences


"""
Helper Functions.
"""

def rot(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def extract_argoverse_sequence_augmented(seq, config, hd_map, train=False, val=False, test=True, augment=False):
    """
    seq: a data sequence from argoverse API
    config: yaml configuration file
    """
    start_time = time.time()

    df = seq.seq_df # Data frame
    city = seq.city # City name

    cur_step = config['current_timestep']
    T_obs = config['observed_horizon']
    T_pred = config['prediction_horizon']
    T_total = T_obs + T_pred
    W = config['kalman_smoother']['process_noise_var']
    V = config['kalman_smoother']['measurement_noise_var']

    # Extract timestamps
    timestamps = np.around(df[df['OBJECT_TYPE'] == 'AGENT']['TIMESTAMP'].to_numpy(), decimals=1)
    timestamp_map = {timestamps[i]: i for i in range(len(timestamps))}

    # Extract agents
    if train or val:
        agents_raw = np.nan * np.zeros((seq.num_tracks,T_total,2), dtype=np.float32)
    elif test:
        agents_raw = np.nan * np.zeros((seq.num_tracks,T_obs,2), dtype=np.float32)

    target_agent_track_id = df[df['OBJECT_TYPE'] == 'AGENT']['TRACK_ID'].values[0]
    agents_raw[0] = seq.agent_traj

    k = 1
    for track_id in seq.track_id_list:
        if track_id == target_agent_track_id:
            continue
        data = df[(df['TRACK_ID'] == track_id) & (df['OBJECT_TYPE'] != 'AGENT')][['TIMESTAMP','X','Y']].to_numpy()
        t = np.around(data[:,0].tolist(), decimals=1)
        i = [timestamp_map[ti] for ti in t]
        agents_raw[k,i] = data[:,1:]
        k += 1

    # Filter out agents with few observed measurements
    valid_entries = np.all(~np.isnan(agents_raw[:,:T_obs]), axis=-1)
    is_valid = np.sum(valid_entries, axis=-1) >= config['min_num_observations']
    agents_raw = agents_raw[is_valid]

    # Local index of possible targets
    targets = [0]
    if augment:
        for k, traj in enumerate(agents_raw):
            if k == 0:
                continue
            valid_entries = np.all(~np.isnan(traj[T_obs:]), axis=-1)

            if np.sum(valid_entries, axis=-1) > config['min_num_future_observations']:
                traj_length = np.linalg.norm(traj[T_obs:][valid_entries][-1] - 
                                             traj[T_obs:][valid_entries][0])
                if traj_length > config['min_augmented_length']:
                    targets.append(k)

    # Kalman smooth raw agent positions to get smooth position and velocities
    agents_full = np.nan * np.zeros((*agents_raw.shape[:2], 4))
    for k, agent_raw in enumerate(agents_raw):
        agents_full[k,:T_obs] = kalman_smoother(agent_raw[:T_obs], T=T_obs, W=W, V=V).numpy()
        if k in targets and not test:
            agents_full[k,T_obs:] = kalman_smoother(agent_raw[T_obs:], T=T_pred, W=W, V=V).numpy()

    data_points = []

    for target in targets:
        origin_location = agents_full[target, cur_step, :2].copy()
        origin_heading = np.arctan2(agents_full[target, cur_step, 3], agents_full[target, cur_step, 2])
        origin = SE2(rot(origin_heading), origin_location)

        # Rotate all agents
        agents_local = np.zeros_like(agents_full, dtype=np.float32)
        for k, agent in enumerate(agents_full):
            agents_local[k,:,:2] = origin.inverse_transform_point_cloud(agent[:,:2]).astype(np.float32)
            agents_local[k,:,2:] = agent[:,2:] @ origin.rotation.astype(np.float32)

        # Put target agent first
        agents_local[[0, target]] = agents_local[[target, 0]]   # CHECK AGAIN THIS LINE
        
        # If augmented target is just a straight trajectory, ignore
        turn = np.abs(np.arctan2(agents_local[0,-1,3], agents_local[0,-1,2]))
        if not target == 0:
            if torch.rand(1) > 0.4:
                # Add only 40% of candidate trajectories
                continue
#           prob_adding = torch.rand(1)
#           if turn < config['min_augmented_turn'] and prob_adding > 0.2:
#               # 20% of the time, augment with trajectories that have no turns
#               continue
#           elif prob_adding < 0.5:
#               # Add 50% of other augmented trajectories (with sufficient turn)
#               continue

        # Extract centerlines and bring to local frame
        observed_traj_length = np.linalg.norm(agents_local[0,cur_step,:2] - agents_local[0,0,:2])
        lane_segment_ids = hd_map.get_lane_ids_in_xy_bbox(
            *origin_location, 
            city, 
            query_search_range_manhattan=max(
                config['centerline_search_range_multiplier'] * observed_traj_length,
                config['centerline_search_range_min'])
        )

        # Driveable areas
        centerlines = np.zeros((len(lane_segment_ids), 10, 2))
        directions = np.zeros((len(lane_segment_ids), 10, 2))
        for i, lane_id in enumerate(lane_segment_ids):
            centerline = hd_map.get_lane_segment_centerline(lane_id, city)[:,:2]
            centerline = origin.inverse_transform_point_cloud(centerline)
            # Interpolate if centerline size is not 10
            if not centerline.shape[0] == 10:
                print('Interpolate from ', centerline.shape[0], ' to 10')
                t = np.linspace(0,0.9,10)
                tp = np.linspace(0,0.9,centerline.shape[0])
                c_interp = np.zeros((10,2))
                c_interp[:,0] = np.interp(t, tp, centerline[:,0])
                c_interp[:,1] = np.interp(t, tp, centerline[:,1])
                centerline = c_interp

            grad = np.gradient(centerline)[0]
            grad /= np.linalg.norm(grad, axis=1).reshape(-1,1)
            centerlines[i] = centerline
            directions[i] = grad

        # Ground truth for target agent
        ground_truth = agents_local[0,T_obs:] if not test else None
        ground_truth_raw = agents_raw[target,T_obs:] if not test else None

        # Create the agents input vector: (A,T,4) that contains x, y, u, v for each agent, starting from target
        agents = agents_local[:,:T_obs]

        # Create the static map input vector: (G,4) that contains x, y, dirx, diry for each lane segment
        static_map = np.concatenate((centerlines, directions), axis=-1).astype(np.float32)

        data_points.append({
            'ground_truth': None if test else ground_truth,
            'ground_truth_raw': None if test else ground_truth_raw,
            'agents': agents,
            'static_map': static_map,
            'origin': origin.transform_matrix.astype(np.float32),
            'city': city,
            'seq_id': int(seq.current_seq.name[:-4]) if target == 0 else None
        })
    
    return data_points

    
def kalman_smoother(y, T=20, W=[1, 1, 1, 1], V=[1, 1], return_accuracy=False):
    """
    The smoother implementation here follows the following paper:
    "Fitting a Kalman Smoother to Data" by Shane Barratt and Stephen Boyd
    link: https://stanford.edu/~boyd/papers/pdf/auto_ks.pdf
    """
    nx = 4
    ny = 2
    dt = 0.1
    N = T * (nx + ny)
    
    if not torch.is_tensor(y):
        y = torch.from_numpy(y).type(torch.float32)
    
    assert y.shape[0] == T
    assert y.shape[1] == 2
    assert len(W) == 4
    assert len(V) == 2
    
    # Dynamics model: single integrator
    A = torch.Tensor([[1, 0, dt, 0], 
                      [0, 1, 0, dt], 
                      [0, 0, 1, 0], 
                      [0, 0, 0, 1]])
    # Measurement model: positions only
    C = torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0]])
    # Process noise covariance
    W = torch.diag(torch.Tensor(W))**2
    # Measurement noise covariance
    V = torch.diag(torch.Tensor(V))**2
    
    # Flatten measurements
    z_map = np.nan * torch.zeros(N)
    z_map[T*nx:] = y.flatten(-2)

    # Selector matrix: weeds out nans in y
    B = torch.eye(N)[~z_map.isnan()]
    
    # Replace nans with -1
    c = y.flatten(-2)
    c = c[~c.isnan()]
    nc = c.shape[-1] # Number of available measurements
    
    # Compute KKT matrix
    Wih = torch.sqrt(W) # This is equivalent to matrix square root, since W is diagonal
    Vih = torch.sqrt(V) # This is equivalent to matrix square root, since V is diagonal
    a = torch.eye(T-1)
    zeros = torch.zeros((T-1)*nx, nx)
    D = torch.zeros((N-nx, N))
    D[:(T-1)*nx, :T*nx] = torch.cat((torch.kron(a, -Wih @ A), zeros), dim=-1) + \
                         torch.cat((zeros, torch.kron(a, Wih)), dim=-1)
    
    a = torch.eye(T)
    D[(T-1)*nx:, :T*nx] = torch.kron(a, -Vih @ C)
    D[(T-1)*nx:, T*nx:] = torch.kron(a, Vih)   
    
    # KKT marix
    M = torch.cat(
    (torch.cat((torch.zeros(N,N), D.T, B.T), dim=-1),
     torch.cat((D, -torch.eye(N-nx), torch.zeros(N-nx,nc)), dim=-1),
     torch.cat((B, torch.zeros(nc,N-nx+nc)), dim=-1)
    ), dim=0)
    
    # Solve using LU decomposition
    f = torch.cat((torch.zeros(2*N-nx), c), dim=0)
    f.unsqueeze_(-1)
    M_LU = torch.lu(M)
    x = torch.lu_solve(f, *M_LU)
    
    if return_accuracy:
        accuracy = torch.linalg.norm(B @ x[:N] - c)
        return x[:T*nx].reshape(-1,nx), accuracy

    return x[:T*nx].reshape(-1,nx)


