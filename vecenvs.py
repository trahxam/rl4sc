from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvStepReturn, VecEnvObs
from gym.spaces import Box
from gym import Wrapper
from typing import Sequence, Optional, List, Union, Any, Type
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from tqdm import tqdm

import time
import numpy as np
import matplotlib.pyplot as plt


class VectorisedSupplyChain(VecEnv):
    def __init__(self, node_df, edge_df, supplies, demands, max_time, num_envs,
                 update_interval=100):
        self.node_df = node_df
        self.edge_df = edge_df
        self.demands = demands
        self.supplies = supplies
        self.max_time = max_time
        self.num_envs = num_envs
        self.update_interval = update_interval

        self.demand_nodes, self.demand_amounts, self.demand_prices = demands
        self.supply_nodes, self.supply_amounts, self.supply_prices = supplies

        node_param_types = {'nodes': int,
                            'node_inv_init': float, 
                            'node_hold_cost': float,
                            'node_inv_capacity': float,
                            'node_pos_x': float,
                            'node_pos_y': float}
        edge_param_types = {'edge_sender': int,
                            'edge_reciever': int,
                            'edge_trans_time': int,
                            'edge_cost': float, 
                            'edge_hold_cost': float, 
                            'edge_prod_yield': float}

        for param_name, param_type in node_param_types.items():
            setattr(self, param_name, self.node_df[param_name].to_numpy(dtype=param_type))

        for param_name, param_type in edge_param_types.items():
            setattr(self, param_name, self.edge_df[param_name].to_numpy(dtype=param_type))

        self.num_nodes = self.node_df.shape[0]
        self.num_edges = self.edge_df.shape[0]

        self.edge_inv_window_size = np.max(self.edge_trans_time) + 1
        self.demand_window_size = 64

        self.action_space_shape = (self.num_edges,)
        self.observation_space_shape = (self.num_nodes + 
                                        self.num_edges * self.edge_inv_window_size + 
                                        len(self.demand_nodes) * self.demand_window_size,)
        
        self.action_space = Box(low=0.0, high=1024, shape=self.action_space_shape, dtype=np.float32)
        self.observation_space = Box(low=0.0, high=np.inf, shape=self.observation_space_shape, dtype=np.float32)
        self.demand_vector = np.hstack([self.demand_amounts, np.zeros(shape=(len(self.demand_nodes), 
                                                                             self.demand_window_size))])
        
        self.action_history = np.zeros(shape=(self.max_time, self.num_edges))
        self.node_profit_history = np.zeros(shape=(self.max_time, self.num_nodes))
        self.node_inv_history = np.zeros(shape=(self.max_time, self.num_nodes))
        self.edge_inv_history = np.zeros(shape=(self.max_time, self.num_edges, self.edge_inv_window_size))
        self.walltime_history = np.zeros(shape=(self.max_time))

        self.batch_node_profit_history = np.zeros(shape=(self.num_envs, self.max_time, self.num_nodes))

        super(VectorisedSupplyChain, self).__init__(self.num_envs, self.observation_space, self.action_space)

    def _get_obs(self):
        # Flatten the node and edge invs, then concat them together

        batch_node_state = self.node_inv
        batch_edge_state = self.edge_inv.reshape(self.num_envs, -1)
        single_demand_state = self.demand_vector[:,self.time:self.time+self.demand_window_size].flatten()
        batch_demand_state = np.repeat(np.expand_dims(single_demand_state, 0), self.num_envs, 0)
        batch_state_vector = np.concatenate([batch_node_state, batch_edge_state, batch_demand_state], axis=1)
        return batch_state_vector
    
    def reset(self):
        self.node_inv = np.zeros(shape=(self.num_envs, self.num_nodes))
        self.edge_inv = np.zeros(shape=(self.num_envs, self.num_edges, self.edge_inv_window_size))
        self.node_profits = np.zeros(shape=(self.num_envs, self.num_nodes))

        # Initialise the node inventories to the specified initialisation values

        self.node_inv = np.repeat(self.node_inv_init[np.newaxis,...], self.num_envs, axis=0)
        self.time = 0

        # Initialise the stuff we will use for logging

        return self._get_obs()
    
    def step(self, actions):

        assert actions.shape == (self.num_envs, self.num_edges)

        tstart = time.time()
        self.node_profits = np.zeros(shape=(self.num_envs, self.num_nodes))
        actions = actions * 10000

        for edge in range(self.num_edges):
            # Get the nodes that are going to do the order

            supplier = self.edge_sender[edge]
            purchaser = self.edge_reciever[edge]

            # Requested stock cannot be bigger than the available stock, and it can't be less than zero

            supplier_stock = self.node_inv[:,supplier]
            requested_stock = np.round(actions[:,edge], 0).astype(np.float32)
            requested_stock = np.maximum(requested_stock, np.zeros_like(requested_stock))
            available_requested_stock = np.minimum(requested_stock, supplier_stock)

            # Requested stock cannot be bigger than the space available in the requesters inventory

            space_available = self.node_inv_capacity[purchaser] - self.node_inv[:,purchaser]
            purchased_stock = np.minimum(available_requested_stock * self.edge_prod_yield[edge], space_available)

            # Move the stock from the supplier node inv to the purchaser edge pipeline

            self.edge_inv[:, edge, self.edge_trans_time[edge]] += purchased_stock
            self.node_inv[:, supplier] -= purchased_stock

            # Settle the cost of purchasing the stock

            stock_cost = purchased_stock * self.edge_cost[edge]
            self.node_profits[:, supplier] += stock_cost 
            self.node_profits[:, purchaser] -= stock_cost 

        for edge in range(self.num_edges):
            sender = self.edge_sender[edge]
            reciever = self.edge_reciever[edge]

            # Move any stock that has arrived from an edge into the purchaser inv and shift the edge along

            available_stock = self.edge_inv[:, edge, 0].copy()
            space_available = self.node_inv_capacity[reciever] - self.node_inv[:, reciever]
            stock = np.minimum(available_stock, space_available).copy()

            self.node_inv[:, reciever] += stock * self.edge_prod_yield[edge]
            self.edge_inv[:, edge, 0] = 0.0
            self.edge_inv[:, edge, :] = np.roll(self.edge_inv[:, edge, :], -1)
            self.edge_inv[:, edge, 0] += (available_stock - stock)

            # Apply edge holding costs

            self.node_profits[:, reciever] -= self.edge_inv[:, edge, 0] * self.edge_hold_cost[edge]

        # Apply node holding costs

        for k in range(self.num_nodes):
            node = self.nodes[k]
            self.node_profits[:, node] -= self.node_inv[:, node] * self.node_hold_cost[node]

        # Realise the market demand
        
        for k in range(self.demand_nodes.shape[0]):
            node = self.demand_nodes[k]
            requested_amount = self.demand_amounts[k, self.time]
            price = self.demand_prices[k, self.time]

            # Purchased demand cannot be more than the available stock

            purchased_amount = np.minimum(requested_amount, self.node_inv[:, node])

            # Take the purchased demand out of the retailer and give them the profit

            self.node_inv[:, node] -= purchased_amount
            self.node_profits[:, node] += purchased_amount * price

        # Realise supply

        for k in range(self.supply_nodes.shape[0]):
            node = self.supply_nodes[k]
            amount = self.supply_amounts[k, self.time]
            price = self.supply_prices[k, self.time]

            # Supply cannot exceed the node capacity

            space_available = self.node_inv_capacity[node] - self.node_inv[:, node]
            amount_added = np.minimum(amount, space_available)

            self.node_inv[:, node] += amount_added
            self.node_profits[:, node] -= amount_added * price

        observation = self._get_obs()
        reward = np.sum(self.node_profits, axis=1)
     
        if self.time >= self.max_time - 1:
            terminated = np.full((self.num_envs), True)
        else:
            terminated = np.full((self.num_envs), False)

        info = {}

        if self.time % self.update_interval == 0:
            mean_tot_reward = np.sum(self.batch_node_profit_history) / (self.num_envs * self.time)
            mean_tot_reward = np.mean(reward)
            print(f'{self.time + self.update_interval}: {mean_tot_reward:.2f}')

        tend = time.time()

        self.batch_node_profit_history[:,self.time,:] = self.node_profits
        self.action_history[self.time,:] = actions[0]
        self.node_profit_history[self.time,:] = self.node_profits[0].copy()
        self.node_inv_history[self.time,:] = self.node_inv[0].copy()
        self.edge_inv_history[self.time,:,:] = self.edge_inv[0].copy()
        self.walltime_history[self.time] = tend - tstart

        self.time += 1

        return observation, reward, terminated, info
    
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
        
    def plot_history(self, linewidth=0.4, alpha=0.6):
        fig, ax = plt.subplots(nrows=6, ncols=1)
        fig.set_size_inches(8, 8)
        args = {'linewidth': linewidth, 'alpha': alpha}

        for edge in range(self.num_edges):
            sender, reciever = self.edge_sender[edge], self.edge_reciever[edge]
            ax[0].plot(self.action_history[:,edge], **args, label=f'Edge {sender}-{reciever}')
            ax[1].plot(self.edge_inv_history[:,edge,0], **args, label=f'Edge {sender}-{reciever}')

        for node in range(self.num_nodes):
            ax[2].plot(self.node_inv_history[:,node], label=f'Node {node}', **args)
            ax[3].plot(self.node_profit_history[:,node], label=f'Node {node}', **args)
            ax[4].plot(np.cumsum(self.node_profit_history[:,node]), label=f'Node {node}', **args)
            #windowed_mean = np.convolve(self.node_profit_history[:,node], np.ones(self.demand_window_size)/self.demand_window_size, mode='valid')
            #ax[4].plot(windowed_mean, label=f'Node {node}', **args)

        for k in range(len(self.demand_nodes)):
            node = self.demand_nodes[k]
            ax[5].plot(self.demand_amounts[k], label=f'Demand {node}', **args)

        for k in range(len(self.demand_nodes)):
            node = self.supply_nodes[k]
            ax[5].plot(self.supply_amounts[k], label=f'Supply {node}', **args)

        axes_labels = ['Actions', 'Edge Stock', 'Node Stock', 'Node Profit', 'Node Cumulative Profit', 'Demand']

        for i in range(6):
            
            ax[i].set_ylabel(axes_labels[i])

            if i == 5:
                ax[i].set_xlabel('Time')
            else:
                ax[i].set_xticks([])
    
        fig.show()

    def get_node_marker(self, node):
        if node in self.demand_nodes:
            return 'o'
        elif node in self.supply_nodes:
            return 'D'
        else:
            return 's'
        
    def plot_chain(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        for edge in range(self.num_edges):
            sender = self.edge_sender[edge]
            reciever = self.edge_reciever[edge]
            ax.plot([self.node_pos_x[sender], self.node_pos_x[reciever]], 
                    [self.node_pos_y[sender], self.node_pos_y[reciever]], color='black', linestyle='--', zorder=0)
            
        for k in range(self.num_nodes):
            node = self.nodes[k]
            marker = self.get_node_marker(node)
            ax.scatter(self.node_pos_x[k], self.node_pos_y[k], ec='black', fc='white', s=400, zorder=10, marker=marker)
            ax.text(self.node_pos_x[k], self.node_pos_y[k], node, zorder=20, verticalalignment='center', horizontalalignment='center')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.show()

    def animate_history(self, time_start, time_end, save_path, node_inv_scale_factor=100):
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 4)

        for edge in range(self.num_edges):
            sender = self.edge_sender[edge]
            reciever = self.edge_reciever[edge]
            ax.plot([self.node_pos_x[sender], self.node_pos_x[reciever]], 
                    [self.node_pos_y[sender], self.node_pos_y[reciever]], color='black', linestyle='--', zorder=0)
            
        for k in range(self.num_nodes):
            node = self.nodes[k]
            marker = self.get_node_marker(node)
            ax.scatter(self.node_pos_x[k], self.node_pos_y[k], ec='black', fc='white', s=400, zorder=10, marker=marker)
            ax.text(self.node_pos_x[k], self.node_pos_y[k], node, zorder=20, verticalalignment='center', horizontalalignment='center')
            bar_height = self.node_inv_capacity[k] * node_inv_scale_factor / np.max(self.node_inv_capacity)
            ax.plot([self.node_pos_x[k], self.node_pos_x[k]], 
                    [self.node_pos_y[k], self.node_pos_y[k] + bar_height], color='lightgray', linewidth=10.0)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

        for boundary_point in [(-100,200), (-100,-200), (500, 200), (500, -200)]:
            boundary_x, boundary_y = boundary_point
            ax.scatter(boundary_x, boundary_y, color='none')

        node_plots = [ax.plot([], [], color='tab:blue', linewidth=10.0)[0] for k in range(self.num_nodes)]
        edge_plots = [ax.plot([], [], color='tab:blue', linewidth=6.0)[0] for k in range(np.sum(self.edge_trans_time) + self.num_edges)]
        time_text = ax.text(-80, 180, '', fontsize=12)

        cmap = cm.get_cmap('RdYlGn')
        
        def animate(time):
            time = time + time_start

            for k in range(self.num_nodes):
                bar_height = self.node_inv_history[time,k] * node_inv_scale_factor / np.max(self.node_inv_capacity)
                
                node_plots[k].set_data([self.node_pos_x[k], self.node_pos_x[k]], 
                                       [self.node_pos_y[k], self.node_pos_y[k] + bar_height])
                node_plots[k].set_color(cmap(self.node_inv_history[time,k] / self.node_inv_capacity[k]))
                
            k = 0
                
            for edge in range(self.num_edges):
                for lead_time in range(self.edge_trans_time[edge]):
                    sender = self.edge_sender[edge]
                    reciever = self.edge_reciever[edge]
                    frac = (lead_time + 1) / (self.edge_trans_time[edge] + 2)
                    base_x = (self.node_pos_x[sender] - self.node_pos_x[reciever]) * frac + self.node_pos_x[reciever]
                    base_y = (self.node_pos_y[sender] - self.node_pos_y[reciever]) * frac + self.node_pos_y[reciever]
                    edge_plots[k].set_data([base_x, base_x], 
                                           [base_y, base_y + self.edge_inv_history[time,edge,lead_time] * node_inv_scale_factor / np.max(self.node_inv_capacity)])
                    k += 1

            time_text.set_text(f'Timestep {time}/{time_end}')

            return node_plots + edge_plots + [time_text]

        anim = FuncAnimation(fig, animate, frames=time_end - time_start, interval=100, blit=True)
        anim.save(save_path, writer='pillow')
        fig.show()
      
