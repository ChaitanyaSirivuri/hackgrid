import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, Birch, OPTICS, AffinityPropagation, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
import pickle
from sklearn.preprocessing import StandardScaler


def evaluate_clustering_model(model, X):
    labels = model.fit_predict(X)

    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    adjusted_rand = adjusted_rand_score(
        labels, labels)  # Dummy labels for clustering
    adjusted_mutual_info = adjusted_mutual_info_score(
        labels, labels)  # Dummy labels for clustering
    # Dummy labels for clustering
    homogeneity = homogeneity_score(labels, labels)
    completeness = completeness_score(
        labels, labels)  # Dummy labels for clustering
    v_measure = v_measure_score(labels, labels)  # Dummy labels for clustering
    fowlkes_mallows = fowlkes_mallows_score(
        labels, labels)  # Dummy labels for clustering

    return silhouette, calinski_harabasz, davies_bouldin, adjusted_rand, adjusted_mutual_info, homogeneity, completeness, v_measure, fowlkes_mallows


def run_clustering_models(models, X):
    results = []

    best_model = None
    best_silhouette = -1.0  # Initialize with a lower value

    for model_name, model in models:
        metrics = evaluate_clustering_model(model, X)
        results.append((model_name,) + metrics)

        # Save the best model based on silhouette score
        if metrics[0] > best_silhouette:
            best_silhouette = metrics[0]
            best_model = model

    return pd.DataFrame(results, columns=["Model", "Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score",
                                          "Adjusted Rand Score", "Adjusted Mutual Info", "Homogeneity", "Completeness",
                                          "V-Measure", "Fowlkes-Mallows"]), best_model


def save_best_model(model, model_name):
    model_filename = f"{model_name}_best_model.pkl"
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Saved the best model to {model_filename}")
    return model_filename


def create_clustering_models():
    return [
        ("K-Means", KMeans(n_clusters=3)),
        ("Agglomerative Clustering", AgglomerativeClustering(n_clusters=3)),
        ("DBSCAN", DBSCAN()),
        ("MeanShift", MeanShift()),
        ("Birch", Birch(n_clusters=3)),
        ("OPTICS", OPTICS()),
        ("Affinity Propagation", AffinityPropagation()),
        ("Mini-Batch K-Means", MiniBatchKMeans(n_clusters=3))
    ]


def start_clustering(data):
    # Exclude the last column from clustering
    X = data.iloc[:, :-1]

    # Perform clustering on the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = create_clustering_models()
    results, best_model = run_clustering_models(models, X_scaled)

    # Sort and reset index for better visualization
    results = results.sort_values(by='Silhouette Score', ascending=False)
    results = results.reset_index(drop=True)
    results.index += 1

    model_filename = save_best_model(best_model, "clustering")
    return results, model_filename

# Example usage with a dataset
# iris = load_iris()
# X = pd.DataFrame(iris.data, columns=iris.feature_names)
# results, model_filename = start_clustering(X)
