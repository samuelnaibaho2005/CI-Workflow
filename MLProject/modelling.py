import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import mlflow
import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings('ignore')

def main(args):
    # Set Experiment (Optional, jika tidak diset akan masuk ke 'Default')
    mlflow.set_experiment("nutrition_clustering_kmeans")

    # Enable MLflow Autolog untuk sklearn
    mlflow.sklearn.autolog()

    # Load preprocessed data
    try:
        df_clean = pd.read_csv("nutrition_preprocessing.csv")
        df_clean = df_clean.drop(columns=['Unnamed: 0'], errors='ignore')
    except FileNotFoundError:
        print("Error: File nutrition_preprocessing.csv tidak ditemukan.")
        return

    # 1. Mencari Nilai K
    print("STEP 1: Mencari Jumlah Cluster Optimal dengan Elbow Method")

    model = KMeans(random_state=args.random_state)
    # Menggunakan range K dari parameter input
    visualizer = KElbowVisualizer(model, k=(args.min_k, args.max_k))
    visualizer.fit(df_clean)
    optimal_k = visualizer.elbow_value_

    print(f"Optimal K (Elbow): {optimal_k}\n")
        
    # 2. K-MEANS CLUSTERING
    print("STEP 2: Training K-Means Model")

    with mlflow.start_run(run_name=f"KMeans_k{optimal_k}"):
        # Train K-Means dengan parameter dari args
        kmeans = KMeans(
            n_clusters=optimal_k,
            random_state=args.random_state,
            n_init=args.n_init,
            max_iter=args.max_iter
        )
        kmeans_labels = kmeans.fit_predict(df_clean)

        # Calculate metrics
        kmeans_silhouette = silhouette_score(df_clean, kmeans_labels)
        kmeans_inertia = kmeans.inertia_
        
        print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
        print(f"K-Means Inertia: {kmeans_inertia:.4f}\n")

    # 4. COMPARISON & ANALYSIS
    print("STEP 4: Perbandingan Model")

    comparison_data = {
        'Algorithm': ['K-Means'],
        'Silhouette Score': [kmeans_silhouette],
        'Inertia': [kmeans_inertia],
        'Clusters': [optimal_k]
    }

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + str(comparison_df))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=10)
    
    args = parser.parse_args()
    
    main(args)