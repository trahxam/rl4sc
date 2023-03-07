import numpy as np
import pandas as pd

from gymnasium.spaces import Box, Graph
from gymnasium import Env

import torch
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, SAGEConv, EdgeConv, GraphSAGE




class GraphSupplyChain(Env):

    def __init__(self, setup_name):
        node_param_types = {'node_inv_init': np.float32, 'node_prod_cost': np.float32, 'node_prod_time': np.int32,
                       'node_hold_cost': np.float32, 'node_prod_capacity': np.float32, 'node_inv_capacity': np.float32}
        edge_param_types = {'edge_trans_time': np.int32, 'edge_cost': np.float32, 
                       'edge_hold_cost': np.float32, 'edge_prod_yield': np.float32}

        self.node_df = pd.read_csv(f'graphsupplychains/{setup_name}/nodes.csv')
        self.edge_df = pd.read_csv(f'graphsupplychains/{setup_name}/edges.csv')

        for param_name, param_type in node_param_types.items():
            setattr(self, param_name, self.node_df[param_name].to_numpy(dtype=param_type))

        for param_name, param_type in edge_param_types.items():
            setattr(self, param_name, self.edge_df[param_name].to_numpy(dtype=param_type))

        self.edge_sender = self.edge_df['edge_sender'].to_numpy()
        self.edge_reciever = self.edge_df['edge_reciever'].to_numpy()

        self.nodes = self.node_df['nodes'].to_numpy()
        self.edges = np.vstack((self.edge_sender, self.edge_reciever))

        self.num_nodes = self.node_df.shape[0]
        self.num_edges = self.edge_df.shape[0]

        self.node_look_ahead_time = 8
        self.edge_look_ahead_time = 16

        self.node_inv = np.zeros(shape=(self.num_nodes, self.node_look_ahead_time))
        self.edge_inv = np.zeros(shape=(self.num_edges, self.edge_look_ahead_time))

        self.node_params = self.node_df[[param for param, _ in node_param_types.items()]].to_numpy()
        self.edge_params = self.edge_df[[param for param, _ in edge_param_types.items()]].to_numpy()

        self.observation_space = Graph(node_space=Box(low=0.0, high=np.inf, shape=(self.node_params.shape[1],), dtype=np.float32),
                                       edge_space=Box(low=0.0, high=np.inf, shape=(self.edge_params.shape[1],), dtype=np.float32))
        
        self.action_space = Box(low=0.0, high=np.inf, shape=(self.edge_params.shape[0],), dtype=np.float32)

        # Define the nodes that will serve demand and the demand time series for each demand node

        self.demand_nodes = np.array([0])
        self.demand_sale_price = np.array([2.2])

        # Define the nodes that will recieve raw supplied and the supply time series for each supply node
    
        self.supply_nodes = np.array([7, 8])
        self.supply_buy_price = np.array([1.0, 1.0])

    def _get_obs(self):
        self.node_features = np.hstack((self.node_params, self.node_inv))
        self.edge_features = np.hstack((self.edge_params, self.edge_inv))

        return Data(x=torch.from_numpy(self.node_features), 
                    edge_index=torch.from_numpy(self.edges),
                    edge_attr=torch.from_numpy(self.edge_features))
    
    def reset(self):
        self.node_inv = np.zeros(shape=(self.num_nodes, self.node_look_ahead_time))
        self.edge_inv = np.zeros(shape=(self.num_edges, self.edge_look_ahead_time))
        self.node_profits = np.zeros(shape=(self.num_nodes))

        # Initialise the node inventories to the specified initialisation values

        self.node_inv[:,0] = self.node_inv_init
        self.time = 0

        self.mean_rewards = []

        return self._get_obs()
    
    def step(self, actions):
        
        assert actions.shape[0] == self.num_edges

        self.node_profits = np.zeros(shape=(self.num_nodes))

        for edge in range(self.num_edges):
            # Get the nodes that are going to do the order

            supplier = self.edges[0, edge]
            purchaser = self.edges[1, edge]

            # Requested stock cannot be bigger than the available stock

            supplier_stock = self.node_inv[supplier, 0]

            requested_stock = np.round(actions[edge], 1).astype(np.float32)
            available_requested_stock = np.min((requested_stock, supplier_stock), axis=0).copy()

            # Requested stock cannot be bigger than the space available in the requesters inventory

            space_available = self.node_inv_capacity[purchaser] - self.node_inv[purchaser, 0]
            purchased_stock = np.min((available_requested_stock * self.edge_prod_yield[edge], space_available), axis=0).copy()

            # Move the stock from the supplier node inv to the purchaser edge pipeline

            self.edge_inv[edge, self.edge_trans_time[edge]] += purchased_stock
            self.node_inv[supplier, 0] -= purchased_stock

            # Settle the cost of purchasing the stock

            stock_cost = purchased_stock * self.edge_cost[edge]

            self.node_profits[supplier] += stock_cost 
            self.node_profits[purchaser] -= stock_cost 

            # Move any stock that has arrived from an edge into the purchaser inv and shift the edge along

            arrived_stock = self.edge_inv[edge, 0].copy()
            self.node_inv[purchaser, self.node_prod_time[purchaser]] += arrived_stock * self.edge_prod_yield[edge]
            self.edge_inv[edge, 0] = 0.0
            self.edge_inv[edge, :] = np.roll(self.edge_inv[edge, :], -1)

        for k in range(self.num_nodes):
            node = self.nodes[k]

            # Move any stock that has completed manufacturing into the node env and shift manufacturing pipeline along

            preexisting_manufactured_stock = self.node_inv[node, 0].copy()

            self.node_inv[node, 0] = 0.0
            self.node_inv[node, :] = np.roll(self.node_inv[node, :], -1)
            new_manufactured_stock = self.node_inv[node, 0].copy()
            self.node_inv[node, 0] += preexisting_manufactured_stock

            # Apply manufacture costs

            self.node_profits[node] -= new_manufactured_stock * self.node_prod_cost[node]

            # Apply node holding costs

            self.node_profits[node] -= self.node_inv[node,0] * self.node_hold_cost[node]

          # Realise the market demand

        for k in range(len(self.demand_nodes)):
            node = self.demand_nodes[k]

            #requested_demand = self.demand[k, self.time].repeat(self.num_envs, axis=0)
            requested_demand = np.random.poisson(lam=10.0)
            #requested_demand = np.full(shape=(self.num_envs), fill_value=10.0)
            purchased_demand = np.min((requested_demand, self.node_inv[node, 0]), axis=0)

            self.node_inv[node, 0] -= purchased_demand
            self.node_profits[node] += purchased_demand * self.demand_sale_price[k]

        # Realise supply

        for k in range(len(self.supply_nodes)):
            node = self.supply_nodes[k]
            supply = np.random.poisson(lam=10.0)

            self.node_inv[node, self.node_prod_time[node]] += supply
            self.node_profits[node] -= supply * self.supply_buy_price[k]

        # Imncrement the time BEFORE we record the state vector so that it includes the demand of the next timestep

        self.time += 1

        observation = self._get_obs()
        reward = np.sum(self.node_profits)
        terminated = False 
        info = {}

        return observation, reward, terminated, info


chain = GraphSupplyChain('test')
chain.reset()
print(chain._get_obs())
action = np.random.poisson(4, chain.num_edges)
print(chain.step(action))
data = chain._get_obs() 

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch.nn import Module

class ConvLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")  
        self.mlp = Sequential(
            Linear(in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        node_concat = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(node_concat)  # shape [num_edges, out_channels]


class GNN(Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, hidden_channels)
        #self.conv2 = ConvLayer(hidden_channels, out_channels)

        self.edge_mlp = Sequential(
            Linear(64, 16),
            ReLU(),
            Linear(16, 1),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        out = self.conv1(x, edge_index, edge_attr)
        #x = self.conv2(x, edge_index, edge_attr)
        src, rec = edge_index

        out_src = out[src]
        out_rec = out[rec]

        return self.edge_mlp(torch.cat([out_src, out_rec], dim=-1))

model = GNN(14 + 14 + 20, 32, 1)


out = model.forward(x=data.x.to(torch.float32), 
                    edge_index=data.edge_index.to(torch.int64), 
                    edge_attr=data.edge_attr.to(torch.float32))

print(out.shape)
print(out)
source, rec = data.edge_index
print(source, source.shape)
print(rec, rec.shape)