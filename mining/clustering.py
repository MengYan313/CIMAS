import pickle
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from skopt import gp_minimize
from skopt.space import Real, Integer
import sys
import csv

def load_pkl(file_path, limit=-1):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
        print(type(data))
        # if isinstance(data, pd.DataFrame):
        print(data.head())
        print(f"type(data):{type(data)}   data.shape:{data.shape}")
        print("type(data['pros_emb'][0][0])", type(data['pros_emb'][0][0]))
        print("data['pros_emb'][0][0].shape", data['pros_emb'][0][0].shape)
        
        if limit != -1: # Only read the first limit elements
            data = data.iloc[:limit]
        print(f"limit:{limit}", data.shape)

    return data


# @deprecated
def perform_dbscan(embeddings, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on embedding vectors.

    Args:
        embeddings (list): List of embedding vectors.
        eps (float): DBSCAN neighborhood distance parameter.
        min_samples (int): Minimum number of samples for DBSCAN.

    Returns:
        array: Cluster labels for the clustering result.
    """
    # Convert to NumPy array
    embeddings_array = np.array(embeddings)
    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = dbscan.fit_predict(embeddings_array)
    return labels


def dense_clustering(data):
    """
    Use the DBSCAN clustering algorithm for code embedding clustering, and select the best clustering parameters
    by optimizing the objective function (maximizing the silhouette score). The function returns the optimized best eps and min_samples parameters.
    """
    data = np.stack([tensor.numpy() for tensor in data]) # Convert embeddings to NumPy array data.shape (8703, 1, 768)
    data = np.squeeze(data) # Use squeeze to remove dimensions of size 1 data.shape (8703, 768)
    print("data.shape", data.shape)
    sys.stdout.flush()
    
    # Define the objective function to minimize (silhouette score)
    def objective(params):
        """
        Define an objective function to search for the best eps neighborhood radius and min_samples minimum sample number in the hyperparameter space.
        The purpose of this objective function is to maximize the silhouette score, so we return the negative silhouette score
        (because the optimizer minimizes the objective function, so we return the negative value to achieve maximization)
        """
        eps, min_samples = params
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data) # Perform DBSCAN clustering algorithm and return the cluster label for each sample (i.e., the cluster number each sample belongs to)
        score = silhouette_score(data, labels) # Calculate the silhouette score of the clustering result
        # print('labels', labels)
        return -score
    
    # Define the parameter space, specify the value range of eps and min_samples
    space = [Real(0.01, 0.5, name='eps'), # Define the value range of the eps parameter as 0.01 to 0.5, which is the hyperparameter that determines the neighborhood size in DBSCAN. Real means the value is a real number
         Integer(2, 10, name='min_samples')] # Define the value range of the min_samples parameter as 2 to 10, which is the hyperparameter that determines the minimum number of neighborhood samples for each core point in DBSCAN
    # Use Bayesian optimization (gp_minimize) to optimize the objective function objective to find the best eps and min_samples parameters
    result = gp_minimize(objective, space, n_calls=50, random_state=0)
    best_eps, best_min_samples = result.x # Get the best hyperparameters eps and min_samples
    best_score = -result.fun # In the objective function, we return the negative silhouette score, so -result.fun is the positive value of the optimal silhouette score

    print('result', result)
    print('best_eps', best_eps)
    print('best_min_samples', best_min_samples)
    print('best_score', best_score)
    sys.stdout.flush()
    return best_eps, best_min_samples # The function returns the optimized best eps and min_samples parameters


def perform_best_DBSCAN_and_analysis(pro_name, pro_src, pro_emb, pro_info, best_eps=0.5, best_min_samples=2):
    """Perform DBSCAN clustering, calculate and analyze the center point and size of each cluster, and the sample closest to the center point (i.e., frequent sub-idiom)"""
    data = np.stack([tensor.numpy() for tensor in pro_emb])  # Store the embedding vector of each sub-idiom
    data = np.squeeze(data)  # Use squeeze to remove dimensions of size 1 data.shape (8703, 768)
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)  # Initialize DBSCAN clustering algorithm with optimal parameters
    cluster_labels = dbscan.fit_predict(data)  # Use DBSCAN clustering; the cluster label each sample belongs to (-1 means noise point)
    unique_labels = set(cluster_labels)  # Get all cluster labels (which categories there are)
    cluster_data = pd.DataFrame(columns=['label', 'center_point', 'else_point', 'cluster_size', 'center_point_info', 'infos', 'loc_label'])

    for label in unique_labels:  # Traverse each cluster category and build cluster_data in turn
        if label == -1:  # Skip noise points
            continue
        points_in_cluster = data[cluster_labels == label]  # Extract sub-idioms belonging to the current cluster label
        # According to the index in cluster_labels that belongs to the current cluster label, extract the corresponding additional information from info_list
        srcs = [pro_src[i] for i in np.where(cluster_labels == label)[0].tolist()]
        infos = [pro_info[i] for i in np.where(cluster_labels == label)[0].tolist()]

        # Calculate the centroid of the current cluster, np.mean calculates the mean of all points along the sample dimension (axis=0) to get the center point of the cluster
        centroid = np.mean(points_in_cluster, axis=0)
        # Calculate the distance between each sample and the cluster center point, and find the sample closest to the center. Return the index and distance of the closest sample (frequent sub-idiom)
        closest_point_idx, _ = pairwise_distances_argmin_min(points_in_cluster, np.array([centroid]))
        if len(closest_point_idx) > 0 and infos is not None:
            # Extract the info_list tuple (project name, file name, function name, code snippet) of the frequent sub-idiom
            closest_point = [infos[idx] for idx in closest_point_idx][0]  # This list has only one tuple [0]
            if closest_point is not None:
                closest_point_str = [srcs[idx] for idx in closest_point_idx][0]  # Get the code snippet of the frequent sub-idiom
                loc1 = closest_point[0]  # Project name
                loc2 = closest_point[1]  # File name
                loc3 = closest_point[2]  # Code segment location
                loc_label = f"{loc1}-{loc2}-{loc3}"
                # Get other code segments except the center node
                else_point = [src for i, src in enumerate(srcs) if i not in closest_point_idx]
            else:
                closest_point_str = None
                else_point = []
        else:
            closest_point = None
            closest_point_str = None
            else_point = []

        print(f"cluster {label}, len {len(points_in_cluster)}: {repr(closest_point_str)}")
        sys.stdout.flush()
        # Add the result of each cluster to the cluster_data DataFrame (cluster label, frequent sub-idiom code snippet, cluster size, frequent sub-idiom info, all sub-idiom info, location label)
        assert len(points_in_cluster) == len(infos) == len(srcs), "The lengths of the three elements are not equal"
        cluster_data.loc[len(cluster_data.index)] = [label, closest_point_str, else_point, len(points_in_cluster), infos[closest_point_idx[0]], infos, loc_label]

    # cluster_data_file = "../files/clustering_result/" + pro_name + ".pkl"
    # pd.to_pickle(cluster_data, cluster_data_file)  # Store the clustering result of a single project
    return cluster_labels, cluster_data


def clustering(pro_name, pro_src, pro_emb, pro_info):
    """Load embedding data and perform clustering operations, and finally call the analysis of clustering results"""
    # embeddings_list = pro_emb # Store the embedding vector of each sub-idiom
    # info_list = pro_info # Store (pro_name, file_name, node_info(dict)) for each sub-idiom

    # best_eps, best_min_samples = dense_clustering(pro_emb) # Get the best clustering hyperparameters (0.5 2)
    # Perform clustering and analysis at the granularity of sub-idioms
    cluster_labels, cluster_data = perform_best_DBSCAN_and_analysis(pro_name, pro_src, pro_emb, pro_info)
    return cluster_data


def process_projects(data):
    """Perform internal clustering on the embedding vectors of each project and output the clustering results"""
    pros_name = data['pros_name']
    pros_src = data['pros_src']
    pros_emb = data['pros_emb']
    pros_info = data['pros_info']
    cluster_results = []

    for i, pro_name in enumerate(pros_name):
        print(f"\nProcessing project: {pro_name}")
        sys.stdout.flush()
        pro_src = pros_src[i]
        pro_emb = pros_emb[i]
        pro_info = pros_info[i]
        if "ToBeDetermined" in pro_src: # Filter out some code fragments that are not idioms
            continue

        cluster_data = clustering(pro_name, pro_src, pro_emb, pro_info)
        # Save the clustering result of the current project
        cluster_results.append({
            "pros_name": pro_name,
            "clusters": cluster_data # Each element is (cluster label, frequent sub-idiom code snippet, other sub-idiom code snippets, cluster size, frequent sub-idiom info, all sub-idiom info, location label)
        })

        # Statistics of clustering information
        num_clusters = len(cluster_data['label'].unique())
        avg_cluster_size = cluster_data['cluster_size'].mean()
        top100_clusters = cluster_data.nlargest(100, 'cluster_size')

        print(f"Clustering info for project {pro_name}:")
        print(f"Number of clusters: {num_clusters}")
        print(f"Average cluster size: {avg_cluster_size}")
        print("Top 100 idioms:")
        for idx, row in top100_clusters.iterrows():
            print(f"cluster {row['label']} - size: {row['cluster_size']} - : {row['center_point']}")
        
        # Save clustering information to CSV file
        csv_file = "files/clusters_top100.csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Number of clusters", "Average cluster size"])
            writer.writerow([num_clusters, avg_cluster_size])
            writer.writerow(["Top 100 idioms"])
            writer.writerow(["Cluster label", "Cluster size", "Code snippet"])
            for idx, row in top100_clusters.iterrows():
                writer.writerow([row['label'], row['cluster_size'], row['center_point']])


    return cluster_results


def save_cluster_results(cluster_results, output_path):
    """Store the clustering results of all projects"""
    with open(output_path, "wb") as f:
        pickle.dump(cluster_results, f)
    print(f"Clustering results of all projects have been saved to {output_path}")


def main():
    # Load data
    # input_file = "../files/codebert_embedding.pkl"
    # output_file = "../files/cluster_results.pkl"

    input_file = "../files/graphcodebert_embedding.pkl"
    output_file = "../files/gcb_cluster_results.pkl"

    print("Start loading data")
    sys.stdout.flush()
    data = load_pkl(input_file) # Test 10, limit=10
    print("Data loading completed")
    sys.stdout.flush()

    cluster_results = process_projects(data)
    save_cluster_results(cluster_results, output_file)


def run_clustering(embedding_file, cluster_result_path):
    print("Start loading data")
    sys.stdout.flush()
    data = load_pkl(embedding_file) # Test 10, limit=10
    print("Data loading completed")
    sys.stdout.flush()

    cluster_results = process_projects(data)
    save_cluster_results(cluster_results, cluster_result_path)


# nohup python3 clustering.py > clustering.log 2>&1 &
if __name__ == "__main__": # 1929614
    main()