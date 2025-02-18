import math

# Function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Points and initial centroids
points = {'A1':(2,10), 'A2':(2,5), 'A3':(8,4), 'B1':(5,8), 'B2':(7,5), 'B3':(6,4), 'C1':(1,2), 'C2':(4,9)}
centroid_1 = points['A1']
centroid_2 = points['B1']
centroid_3 = points['C1']

# Number of iterations for convergence
for iteration in range(3):  # Can adjust the number of iterations

    # Initialize clusters
    cluster = {1: [], 2: [], 3: []}

    # Assign points to the nearest centroid
    for point, coords in points.items():
        dist_1 = euclidean_distance(coords, centroid_1)
        dist_2 = euclidean_distance(coords, centroid_2)
        dist_3 = euclidean_distance(coords, centroid_3)

        # Find the nearest centroid
        min_dist = min(dist_1, dist_2, dist_3)

        # Assign point to the nearest centroid's cluster
        if min_dist == dist_1:
            cluster[1].append(coords)
        elif min_dist == dist_2:
            cluster[2].append(coords)
        else:
            cluster[3].append(coords)

    # Update centroids by calculating the mean of the points in each cluster
    def calculate_new_centroid(cluster_points):
        if len(cluster_points) == 0:
            return (0, 0)
        x_coords = [p[0] for p in cluster_points]
        y_coords = [p[1] for p in cluster_points]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    centroid_1 = calculate_new_centroid(cluster[1])
    centroid_2 = calculate_new_centroid(cluster[2])
    centroid_3 = calculate_new_centroid(cluster[3])

    # Output results for each iteration
    print(f"Iteration {iteration + 1}")
    print(f"Cluster 1 (Centroid {centroid_1}): {cluster[1]}")
    print(f"Cluster 2 (Centroid {centroid_2}): {cluster[2]}")
    print(f"Cluster 3 (Centroid {centroid_3}): {cluster[3]}")
    print()