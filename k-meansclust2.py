import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    'Record': ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'],
    'A': [1.0, 1.5, 3.0, 5.0, 3.5, 4.5, 3.5],
    'B': [1.0, 2.0, 4.0, 7.0, 5.0, 5.0, 4.5]
}

def kmeans_clustering(data, initial_centers, iterations=2):
    # Convert data to numpy array for easier calculations
    points = np.array(list(zip(data['A'], data['B'])))
    centers = np.array(initial_centers)
    n_clusters = len(centers)
    
    # Store history of centers and assignments for visualization
    centers_history = [centers.copy()]
    assignments_history = []
    
    for iteration in range(iterations):
        print(f"\nIteration {iteration + 1}")
        
        # Step 1: Assign points to nearest center
        distances = np.array([np.linalg.norm(points - center, axis=1) for center in centers])
        assignments = np.argmin(distances, axis=0)
        assignments_history.append(assignments)
        
        print("\nPoint Assignments:")
        for i, (point, assignment) in enumerate(zip(points, assignments)):
            print(f"R{i+1} {point}: Cluster {assignment + 1}")
        
        # Step 2: Update centers
        new_centers = np.array([points[assignments == i].mean(axis=0) 
                              for i in range(n_clusters)])
        
        print("\nNew Centers:")
        for i, center in enumerate(new_centers):
            print(f"Cluster {i + 1}: {center}")
        
        centers = new_centers
        centers_history.append(centers.copy())
    
    return centers, assignments, centers_history, assignments_history

def plot_clusters(points, assignments, centers_history, assignments_history, iteration):
    plt.figure(figsize=(10, 6))
    colors = ['b', 'r']
    
    # Plot points with their cluster assignments
    for i in range(len(points)):
        plt.scatter(points[i, 0], points[i, 1], 
                   c=colors[assignments_history[iteration][i]], 
                   label=f'Cluster {assignments_history[iteration][i] + 1}')
    
    # Plot centers
    for i, center in enumerate(centers_history[iteration]):
        plt.scatter(center[0], center[1], c=colors[i], marker='x', s=200, 
                   linewidths=3, label=f'Center {i + 1}')
    
    # Plot previous center if not first iteration
    if iteration > 0:
        for i, prev_center in enumerate(centers_history[iteration - 1]):
            plt.scatter(prev_center[0], prev_center[1], c=colors[i], 
                       marker='x', s=100, linewidths=1, alpha=0.3)
    
    plt.title(f'K-means Clustering - Iteration {iteration}')
    plt.xlabel('A')
    plt.ylabel('B')
    plt.grid(True)
    plt.legend()
    plt.show()

# Initial cluster centers
initial_centers = [(1.0, 1.0), (5.0, 7.0)]

# Run K-means clustering
points = np.array(list(zip(data['A'], data['B'])))
final_centers, final_assignments, centers_history, assignments_history = kmeans_clustering(data, initial_centers)

# Plot results for each iteration
for i in range(len(assignments_history)):
    plot_clusters(points, final_assignments, centers_history, assignments_history, i)

# Print final results
print("\nFinal Clustering Results:")
for i, (point, assignment) in enumerate(zip(points, final_assignments)):
    print(f"Record R{i+1} {point}: Cluster {assignment + 1}")

print("\nFinal Cluster Centers:")
for i, center in enumerate(final_centers):
    print(f"Cluster {i + 1}: {center}")