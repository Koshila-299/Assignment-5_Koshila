# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 21:06:38 2025

@author: DELL
"""
import numpy as np

dataset1_path = r'C:\Users\DELL\Assignment 5\dataset1.npy'
dataset2_path = r'C:\Users\DELL\Assignment 5\dataset2.npy'

pcd1 = np.load(dataset1_path)
pcd2 = np.load(dataset2_path)

###

print(f"Dataset 1 original shape: {pcd1.shape}")
print(f"Dataset 2 original shape: {pcd2.shape}")

import os

parent_path = r'C:\Users\DELL\Assignment 5'
image_path = os.path.join(parent_path, 'images')

if not os.path.exists(parent_path):
    print(f"Error: Parent directory does not exist: {parent_path}")
else:
    os.makedirs(image_path, exist_ok=True)
    print(f"Directory created or already exists: {image_path}")

###
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


dataset1_path = r'C:\Users\DELL\Assignment 5\dataset1.npy'
dataset2_path = r'C:\Users\DELL\Assignment 5\dataset2.npy'


pcd1 = np.load(dataset1_path)
pcd2 = np.load(dataset2_path)

print(f"Dataset 1 shape: {pcd1.shape}")
print(f"Dataset 2 shape: {pcd2.shape}")


output_dir = r'C:\Users\DELL\Assignment 5\images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Directory created or already exists: {output_dir}")


def get_ground_level(pcd):
    heights = pcd[:, 2]  
    hist, bin_edges = np.histogram(heights, bins=100)
    ground_level = bin_edges[np.argmax(hist)]  
    return ground_level

ground_level1 = get_ground_level(pcd1)
ground_level2 = get_ground_level(pcd2)

print(f"Estimated Ground Level for Dataset 1: {ground_level1}")
print(f"Estimated Ground Level for Dataset 2: {ground_level2}")


def show_cloud(points_plt, ground_level=None, title="Point Cloud", save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_plt[:, 0], points_plt[:, 1], points_plt[:, 2], 
               s=0.01, c='b', label='Points')

    if ground_level is not None:
        x_range = np.linspace(points_plt[:, 0].min(), points_plt[:, 0].max(), 10)
        y_range = np.linspace(points_plt[:, 1].min(), points_plt[:, 1].max(), 10)
        xx, yy = np.meshgrid(x_range, y_range)
        zz = np.full_like(xx, ground_level)
        ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z (Elevation)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")

    plt.show()
    
###


import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator


def compute_k_distances(data, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(data)
    distances, _ = nbrs.kneighbors(data)
    return np.sort(distances[:, k-1])


def apply_dbscan(data, eps_value, min_samples=5):
    db = DBSCAN(eps=eps_value, min_samples=min_samples)
    labels = db.fit_predict(data)
    return labels

def plot_clusters(data, labels, title='DBSCAN Clustering'):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=10)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Cluster ID')
    plt.show()


dataset1_path = r'C:\Users\DELL\Assignment 5\dataset1.npy'
dataset2_path = r'C:\Users\DELL\Assignment 5\dataset2.npy'


pcd1_above_ground = np.load(dataset1_path)
pcd2_above_ground = np.load(dataset2_path)


image_path = r'C:\Users\DELL\Assignment 5\plots'
os.makedirs(image_path, exist_ok=True)


k = 5
distances1 = compute_k_distances(pcd1_above_ground, k)
distances2 = compute_k_distances(pcd2_above_ground, k)


plt.figure(figsize=(10, 6))
plt.plot(distances1, label='k-distance')
plt.title('Elbow Plot for Dataset 1 (k=5)')
plt.xlabel('Points (sorted)')
plt.ylabel('Distance')
plt.legend()
plt.savefig(os.path.join(image_path, 'elbow_curve_dataset1.png')) 
plt.show()  
plt.close()  


plt.figure(figsize=(10, 6))
plt.plot(distances2, label='k-distance')
plt.title('Elbow Plot for Dataset 2 (k=5)')
plt.xlabel('Points (sorted)')
plt.ylabel('Distance')
plt.legend()
plt.savefig(os.path.join(image_path, 'elbow_curve_dataset2.png'))  
plt.show()  
plt.close()  


knee1 = KneeLocator(range(len(distances1)), distances1, curve='convex', direction='increasing')
optimal_eps1 = distances1[knee1.knee] if knee1.knee is not None else 0.5
print(f"Optimal eps for Dataset 1: {optimal_eps1}")

knee2 = KneeLocator(range(len(distances2)), distances2, curve='convex', direction='increasing')
optimal_eps2 = distances2[knee2.knee] if knee2.knee is not None else 0.5
print(f"Optimal eps for Dataset 2: {optimal_eps2}")


labels1 = apply_dbscan(pcd1_above_ground, optimal_eps1)
labels2 = apply_dbscan(pcd2_above_ground, optimal_eps2)


plot_clusters(pcd1_above_ground, labels1, title=f'DBSCAN Clustering for Dataset 1 (eps={optimal_eps1})')


plot_clusters(pcd2_above_ground, labels2, title=f'DBSCAN Clustering for Dataset 2 (eps={optimal_eps2})')

###
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN


def apply_dbscan(data, eps_value, min_samples=5):
    db = DBSCAN(eps=eps_value, min_samples=min_samples)
    labels = db.fit_predict(data)
    return labels

def find_largest_cluster(data, labels):
    unique_labels = set(labels)
    largest_cluster = None
    largest_cluster_size = 0
    
    for label in unique_labels:
        if label == -1:  
            continue
        
       
        cluster_points = data[labels == label]
        
      
        if len(cluster_points) > largest_cluster_size:
            largest_cluster_size = len(cluster_points)
            largest_cluster = cluster_points
    
    return largest_cluster


def calculate_span(cluster):
    x_span = cluster[:, 0].max() - cluster[:, 0].min()
    y_span = cluster[:, 1].max() - cluster[:, 1].min()
    return x_span, y_span


def plot_and_calculate_spans(catenary_points, dataset_name):
    if len(catenary_points) > 0:
        min_x = np.min(catenary_points[:, 0])
        max_x = np.max(catenary_points[:, 0])
        min_y = np.min(catenary_points[:, 1])
        max_y = np.max(catenary_points[:, 1])
        x_span = max_x - min_x
        y_span = max_y - min_y
        print(f"{dataset_name} Catenary - Min X: {min_x}, Max X: {max_x}, Min Y: {min_y}, Max Y: {max_y}")
        print(f"{dataset_name} Catenary - X Span: {x_span}, Y Span: {y_span}")
        
        plt.figure(figsize=(10, 10))
        plt.scatter(catenary_points[:, 0], catenary_points[:, 1], s=2, c='#800000', label='Catenary Clusters')
        plt.title(f'{dataset_name}: Catenary Clusters (Linear Structures)', fontsize=20)
        plt.xlabel('X axis', fontsize=14)
        plt.ylabel('Y axis', fontsize=14)
        plt.legend()
        plt.show()  


def plot_3d_catenary(catenary_points, dataset_name):
    if len(catenary_points) > 0:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(catenary_points[:, 0], catenary_points[:, 1], catenary_points[:, 2], 
                             c=catenary_points[:, 2], cmap='viridis', s=2)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.set_title(f'{dataset_name}: 3D Catenary Clusters', fontsize=20)
        plt.colorbar(scatter, label='Z-value')
        plt.show()  
    else:
        print(f"No catenary points to plot for {dataset_name}")


def cluster_and_filter_catenary(data, eps_value, ground_level, dataset_name):

    labels = apply_dbscan(data, eps_value)

  
    largest_cluster = find_largest_cluster(data, labels)

    
    if largest_cluster is not None:
        catenary_points = largest_cluster
    else:
        catenary_points = np.array([])

    return catenary_points, labels


dataset1_path = r'C:\Users\DELL\Assignment 5\dataset1.npy'
dataset2_path = r'C:\Users\DELL\Assignment 5\dataset2.npy'


pcd1_above_ground = np.load(dataset1_path)
pcd2_above_ground = np.load(dataset2_path)


est_ground_level1 = 0.0  
est_ground_level2 = 0.0  


optimal_eps1 = 0.5  
optimal_eps2 = 0.5  


catenary_points1, labels1 = cluster_and_filter_catenary(pcd1_above_ground, optimal_eps1, est_ground_level1, "Dataset 1")
plot_and_calculate_spans(catenary_points1, "Dataset 1")


catenary_points2, labels2 = cluster_and_filter_catenary(pcd2_above_ground, optimal_eps2, est_ground_level2, "Dataset 2")
plot_and_calculate_spans(catenary_points2, "Dataset 2")


plot_3d_catenary(catenary_points1, "Dataset 1")


plot_3d_catenary(catenary_points2, "Dataset 2")

