
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvStepReturn, VecEnvObs
from gym import Wrapper
from gym.spaces import Box
from typing import Sequence, Optional, List, Union, Any, Type
from scipy.stats import poisson


import numpy as np
import pandas as pd
import os


class VectorisedSupplyChain(VecEnv):
  def __init__(self, num_envs, setup_name='test'):
    self.num_envs = num_envs
    
    node_params = {'nodes': np.int32, 'node_inv_init': np.float32, 'node_prod_cost': np.float32, 'node_prod_time': np.int32, 
                    'node_hold_cost': np.float32, 'node_prod_capacity': np.float32, 'node_inv_capacity': np.float32}
    edge_params = {'edge_sender': np.int32, 'edge_reciever': np.int32, 'edge_trans_time': np.int32,
                    'edge_cost': np.float32, 'edge_hold_cost': np.float32, 'edge_prod_yield': np.float32}

    self.node_df = pd.read_csv(f'vecsupplychains/{setup_name}/nodes.csv')
    self.edge_df = pd.read_csv(f'vecsupplychains/{setup_name}/edges.csv')

    print(f'\nLoaded setup: {setup_name}:\n')

    for param_name, param_type in node_params.items():
        setattr(self, param_name, self.node_df[param_name].to_numpy(dtype=param_type))

    for param_name, param_type in edge_params.items():
        setattr(self, param_name, self.edge_df[param_name].to_numpy(dtype=param_type))

    self.edges = np.vstack((self.edge_sender, self.edge_reciever))
    self.num_nodes = self.nodes.shape[0]
    self.num_edges = self.edges.shape[1]

    self.node_look_ahead_time = 8
    self.edge_look_ahead_time = 16

    self.time = 0
    self.max_time_length = int(1e5)

    # Define the nodes that will serve demand and the demand time series for each demand node

    self.demand_nodes = np.array([0])
    self.demand = np.random.poisson(lam=10, size=(len(self.demand_nodes), self.max_time_length))
    self.demand_sale_price = np.array([2.2])

    self.demand_fns = {0: poisson(10 + 10*np.sin(2*self.time*np.pi/7) + 40*np.sin(2*self.time*np.pi/365))}

    # Define the nodes that will recieve raw supplied and the supply time series for each supply node
    
    self.supply_nodes = np.array([7, 8])
    self.supply = np.random.poisson(lam=10, size=(len(self.supply_nodes), self.max_time_length))
    self.supply_buy_price = np.array([1.0, 1.0])

    # Define the action and observation spaces for stable baselines

    self.action_space = Box(low=0.0, high=np.inf, shape=(self.num_edges,), dtype=np.float32)
    self.observation_space = Box(low=0.0, high=np.inf, shape=(len(self.demand_nodes) + 
                                                              self.num_nodes * self.node_look_ahead_time + 
                                                              self.num_edges * self.edge_look_ahead_time,), dtype=np.float32)

    super(VectorisedSupplyChain, self).__init__(self.num_envs, self.observation_space, self.action_space)

  def get_state_vector(self) -> np.ndarray:
    demand_state = np.expand_dims(self.demand[:,self.time].repeat(self.num_envs, axis=0), 1)
    node_state = self.node_inv.reshape(self.num_envs, self.num_nodes * self.node_look_ahead_time)
    edge_state = self.edge_inv.reshape(self.num_envs, self.num_edges * self.edge_look_ahead_time)
    state_vector = np.concatenate([demand_state, node_state, edge_state], axis=1)
    return state_vector

  def reset(self) -> VecEnvObs:
    self.node_inv = np.zeros(shape=(self.num_envs, self.num_nodes, self.node_look_ahead_time))
    self.edge_inv = np.zeros(shape=(self.num_envs, self.num_edges, self.edge_look_ahead_time))
    self.node_profits = np.zeros(shape=(self.num_envs, self.num_nodes))

    # Initialise the node inventories to the specified initialisation values

    self.node_inv[:,:,0] = np.repeat(self.node_inv_init[np.newaxis,...], self.num_envs, axis=0)
    self.time = 0

    self.mean_rewards = []

    return self.get_state_vector()
  
  def step(self, actions: np.ndarray) -> VecEnvStepReturn:
    self.node_profits = np.zeros(shape=(self.num_envs, self.num_nodes))

    for edge in range(self.num_edges):
      # Get the nodes that are going to do the order

      supplier = self.edges[0, edge]
      purchaser = self.edges[1, edge]

      # Requested stock cannot be bigger than the available stock

      supplier_stock = self.node_inv[:, supplier, 0]

      requested_stock = np.round(actions[:,edge], 1).astype(np.float32)
      available_requested_stock = np.min((requested_stock, supplier_stock), axis=0).copy()

      # Requested stock cannot be bigger than the space available in the requesters inventory

      space_available = self.node_inv_capacity[purchaser] - self.node_inv[:, purchaser, 0]
      purchased_stock = np.min((available_requested_stock * self.edge_prod_yield[edge], space_available), axis=0).copy()

      # Move the stock from the supplier node inv to the purchaser edge pipeline

      self.edge_inv[:, edge, self.edge_trans_time[edge]] += purchased_stock
      self.node_inv[:, supplier, 0] -= purchased_stock

      # Settle the cost of purchasing the stock

      stock_cost = purchased_stock * self.edge_cost[edge]

      self.node_profits[:, supplier] += stock_cost 
      self.node_profits[:, purchaser] -= stock_cost 

      # Move any stock that has arrived from an edge into the purchaser inv and shift the edge along

      arrived_stock = self.edge_inv[:, edge, 0].copy()
      self.node_inv[:, purchaser, self.node_prod_time[purchaser]] += arrived_stock * self.edge_prod_yield[edge]
      self.edge_inv[:, edge, 0] = 0.0
      self.edge_inv[:, edge, :] = np.roll(self.edge_inv[:, edge, :], -1)

    for k in range(self.num_nodes):
      node = self.nodes[k]

      # Move any stock that has completed manufacturing into the node env and shift manufacturing pipeline along

      preexisting_manufactured_stock = self.node_inv[:, node, 0].copy()

      self.node_inv[:, node, 0] = 0.0
      self.node_inv[:, node, :] = np.roll(self.node_inv[:, node, :], -1)
      new_manufactured_stock = self.node_inv[:, node, 0].copy()
      self.node_inv[:, node, 0] += preexisting_manufactured_stock

      # Apply manufacture costs

      self.node_profits[:, node] -= new_manufactured_stock * self.node_prod_cost[node]

      # Apply node holding costs

      self.node_profits[:, node] -= self.node_inv[:, node,0] * self.node_hold_cost[node]

    # Realise the market demand

    for k in range(len(self.demand_nodes)):
      node = self.demand_nodes[k]

      #requested_demand = self.demand[k, self.time].repeat(self.num_envs, axis=0)
      requested_demand = np.random.poisson(lam=10.0, size=(self.num_envs))
      #requested_demand = np.full(shape=(self.num_envs), fill_value=10.0)
      purchased_demand = np.min((requested_demand, self.node_inv[:, node, 0]), axis=0)

      self.node_inv[:, node, 0] -= purchased_demand
      self.node_profits[:, node] += purchased_demand * self.demand_sale_price[k]

    # Realise supply

    for k in range(len(self.supply_nodes)):
      node = self.supply_nodes[k]
      self.node_inv[:, node, self.node_prod_time[node]] = 10.0

      #self.node_inv[:, node, self.node_prod_time[node]] = self.supply[k, self.time].repeat(self.num_envs, axis=0)
      #self.node_profits[:, node] -= self.supply[k, self.time] * self.supply_buy_price[k]

    # Imncrement the time BEFORE we record the state vector so that it includes the demand of the next timestep

    self.time += 1

    observation = self.get_state_vector()
    reward = np.sum(self.node_profits, axis=1)
    
    if self.time >= self.max_time_length:
      done = np.full((self.num_envs,), True, dtype=np.bool_)
    else:
      done = np.full((self.num_envs,), False, dtype=np.bool_)

    info = {}

    self.mean_rewards.append(np.mean(reward))

    return observation, reward, done, info
  
  def close(self) -> None:
    return super().close()
  
  def env_is_wrapped(self, wrapper_class: Type[Wrapper], indices: VecEnvIndices = None) -> List[bool]:
    return super().env_is_wrapped(wrapper_class, indices)
  
  def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
    return super().env_method(method_name, *method_args, indices=indices, **method_kwargs)
  
  def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
    return super().get_attr(attr_name, indices)
  
  def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
    return super().seed(seed)
  
  def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
    return super().set_attr(attr_name, value, indices)
  
  def step_async(self, actions: np.ndarray) -> None:
    return super().step_async(actions)
  
  def step_wait(self) -> VecEnvStepReturn:
    return super().step_wait()
  
