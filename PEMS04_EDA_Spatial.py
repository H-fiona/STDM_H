import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import W
from esda.moran import Moran
import os
import networkx as nx

# Create the EDA_Results directory if it doesn't exist
if not os.path.exists('EDA_Results'):
    os.makedirs('EDA_Results')

# Load PEMS04 data from a .npz file
pems_data = np.load('data/pems04.npz')['data']  # Shape: (16992, 307, 3)
flow_data_all = pems_data[:, :, 0]  # Extract flow data for all sensors (16992, 307)

# Load adjacency matrix data from a CSV file
adj_data = pd.read_csv('data/PEMS04.csv')
print(adj_data.head())  # Display the first few rows to check data format

# Build the distance matrix
n_sensors = 307  # Total number of sensors
distance_matrix = np.full((n_sensors, n_sensors), np.inf)  # Initialize with infinity
np.fill_diagonal(distance_matrix, 0)  # Set diagonal to 0 (distance to self)

# Populate the distance matrix with costs from adjacency data
for _, row in adj_data.iterrows():
    from_id = int(row['from'])
    to_id = int(row['to'])
    cost = row['cost']
    distance_matrix[from_id, to_id] = cost
    distance_matrix[to_id, from_id] = cost  # Ensure symmetry

# Build the weight matrix (based on distance threshold)
# Convert distances to weights (weight = 1/distance, weight = 0 for infinite distance)
weight_matrix = np.zeros_like(distance_matrix)
non_inf_mask = distance_matrix != np.inf  # Mask for non-infinite distances
weight_matrix[non_inf_mask] = 1 / distance_matrix[non_inf_mask]  # Assign weights as inverse of distance

# Create a spatial weights object using libpysal
w_dict = {}
for i in range(n_sensors):
    neighbors = np.where(weight_matrix[i, :] > 0)[0].tolist()  # Find indices of non-zero weights
    weights = weight_matrix[i, neighbors].tolist()  # Get corresponding weights
    w_dict[i] = dict(zip(neighbors, weights))  # Map neighbors to weights
w = W(w_dict)  # Create spatial weights object

# Calculate Moran's I using the average flow of the first day
flow_day1 = flow_data_all[:288, :].mean(axis=0)  # Average flow for the first day (307 sensors, 288 time steps = 1 day)
moran = Moran(flow_day1, w)  # Compute Moran's I for spatial autocorrelation
print(f"Moran's I: {moran.I:.4f}")  # Print Moran's I value
print(f"p-value: {moran.p_sim:.4f}")  # Print p-value from permutation test

# Save Moran's I results to a text file
with open('EDA_Results/moran_results.txt', 'w') as f:
    f.write(f"Moran's I: {moran.I:.4f}\n")
    f.write(f"p-value: {moran.p_sim:.4f}\n")

# Plot a histogram of the traffic flow distribution
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
plt.hist(flow_day1, bins=30, color='#1f77b4', alpha=0.7)  # Create histogram with 30 bins, blue color, 70% opacity
plt.title('Distribution of Average Traffic Flow (Day 1)')  # Set plot title
plt.xlabel('Average Traffic Flow')  # Label x-axis
plt.ylabel('Frequency')  # Label y-axis
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('EDA_Results/flow_distribution_histogram.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Build a network graph using NetworkX
G = nx.Graph()  # Initialize an undirected graph
for _, row in adj_data.iterrows():
    from_id = int(row['from'])
    to_id = int(row['to'])
    cost = row['cost']
    G.add_edge(from_id, to_id, weight=cost)  # Add edge with weight based on cost

# Draw the network graph
plt.figure(figsize=(10, 8))  # Set figure size to 10x8 inches
pos = nx.spring_layout(G)  # Use spring layout for node positioning
nx.draw(G, pos, with_labels=True, node_size=50, node_color='#1f77b4', font_size=8, edge_color='gray')  # Draw graph
plt.title('Network Graph of Sensor Connections')  # Set plot title
plt.savefig('EDA_Results/network_graph.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot