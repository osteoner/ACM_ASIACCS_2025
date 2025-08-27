import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, 
    calinski_harabasz_score, 
    davies_bouldin_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    v_measure_score
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import defaultdict, Counter
import logging
from typing import List, Dict, Tuple, Optional, Union
import warnings

warnings.filterwarnings('ignore')

class ClusterEvaluator:

    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, 
                 texts: Optional[List[str]] = None):
        
        self.embeddings = embeddings
        self.labels = labels
        self.texts = texts
        self.n_samples = len(embeddings)
        self.unique_labels = np.unique(labels)
        self.n_clusters = len(self.unique_labels[self.unique_labels != -1])  # Exclude noise (-1)
        
        # Pre-compute cluster assignments
        self.label2docs = defaultdict(list)
        for i, label in enumerate(labels):
            self.label2docs[label].append(i)
    
    def silhouette_score_metric(self) -> float:
        try:
            # Exclude noise points (-1) for silhouette calculation
            mask = self.labels != -1
            if np.sum(mask) < 2 or len(np.unique(self.labels[mask])) < 2:
                return -1.0
            
            score = silhouette_score(self.embeddings[mask], self.labels[mask])
            return float(score)
        except Exception as e:
            logging.warning(f"Error calculating silhouette score: {e}")
            return -1.0
    
    def calinski_harabasz_index(self) -> float:
        try:
            # Exclude noise points (-1)
            mask = self.labels != -1
            if np.sum(mask) < 2 or len(np.unique(self.labels[mask])) < 2:
                return 0.0
                
            score = calinski_harabasz_score(self.embeddings[mask], self.labels[mask])
            return float(score)
        except Exception as e:
            logging.warning(f"Error calculating Calinski-Harabasz index: {e}")
            return 0.0
    
    def davies_bouldin_score_metric(self) -> float:
        try:
            # Exclude noise points (-1)
            mask = self.labels != -1
            if np.sum(mask) < 2 or len(np.unique(self.labels[mask])) < 2:
                return float('inf')
                
            score = davies_bouldin_score(self.embeddings[mask], self.labels[mask])
            return float(score)
        except Exception as e:
            logging.warning(f"Error calculating Davies-Bouldin score: {e}")
            return float('inf')
    
    def dunn_index(self) -> float:
        try:
            from sklearn.metrics.pairwise import pairwise_distances
            
            # Filter out noise points
            valid_labels = [label for label in self.unique_labels if label != -1]
            if len(valid_labels) < 2:
                return 0.0
            
            mask = self.labels != -1
            filtered_embeddings = self.embeddings[mask]
            filtered_labels = self.labels[mask]
            
            # Calculate all pairwise distances at once
            distances = pairwise_distances(filtered_embeddings)
            
            # Calculate minimum inter-cluster distance
            min_inter_distance = float('inf')
            for i, label1 in enumerate(valid_labels):
                for label2 in valid_labels:
                    if label1 >= label2:
                        continue
                    
                    mask1 = filtered_labels == label1
                    mask2 = filtered_labels == label2
                    
                    inter_distances = distances[np.ix_(mask1, mask2)]
                    min_inter_distance = min(min_inter_distance, np.min(inter_distances))
            
            # Calculate maximum intra-cluster distance (diameter)
            max_intra_distance = 0.0
            for label in valid_labels:
                mask_label = filtered_labels == label
                if np.sum(mask_label) > 1:
                    intra_distances = distances[np.ix_(mask_label, mask_label)]
                    max_diameter = np.max(intra_distances)
                    max_intra_distance = max(max_intra_distance, max_diameter)
            
            if max_intra_distance == 0:
                return float('inf') if min_inter_distance > 0 else 0.0
                
            return min_inter_distance / max_intra_distance
            
        except Exception as e:
            logging.warning(f"Error calculating Dunn index: {e}")
            return 0.0
    
    def semantic_coherence(self) -> float:
        try:
            # Filter out noise points
            mask = self.labels != -1
            if np.sum(mask) < 2:
                return 0.0
                
            filtered_embeddings = self.embeddings[mask]
            filtered_labels = self.labels[mask]
            valid_labels = np.unique(filtered_labels)
            
            if len(valid_labels) < 2:
                return 0.0
            
            # Calculate all pairwise cosine similarities at once
            similarity_matrix = cosine_similarity(filtered_embeddings)
            
            # Calculate within-cluster similarities
            within_cluster_sims = []
            for label in valid_labels:
                label_mask = filtered_labels == label
                label_indices = np.where(label_mask)[0]
                
                if len(label_indices) > 1:
                    # Get similarities within this cluster
                    cluster_sims = similarity_matrix[np.ix_(label_indices, label_indices)]
                    # Get upper triangle (excluding diagonal)
                    triu_indices = np.triu_indices_from(cluster_sims, k=1)
                    within_cluster_sims.extend(cluster_sims[triu_indices])
            
            # Calculate between-cluster similarities
            between_cluster_sims = []
            for i, label1 in enumerate(valid_labels):
                for label2 in valid_labels[i+1:]:
                    mask1 = filtered_labels == label1
                    mask2 = filtered_labels == label2
                    indices1 = np.where(mask1)[0]
                    indices2 = np.where(mask2)[0]
                    
                    between_sims = similarity_matrix[np.ix_(indices1, indices2)]
                    between_cluster_sims.extend(between_sims.flatten())
            
            if not within_cluster_sims or not between_cluster_sims:
                return 0.0
            
            avg_within = np.mean(within_cluster_sims)
            avg_between = np.mean(between_cluster_sims)
            
            return avg_within - avg_between
            
        except Exception as e:
            logging.warning(f"Error calculating semantic coherence: {e}")
            return 0.0
    
    def cluster_stability(self, sample_ratio: float = 0.8, n_iterations: int = 10) -> float:

        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics import adjusted_rand_score
            
            stability_scores = []
            n_samples = int(len(self.embeddings) * sample_ratio)
            
            for _ in range(n_iterations):
                indices = np.random.choice(len(self.embeddings), n_samples, replace=False)
                sample_embeddings = self.embeddings[indices]
                

                clusterer = DBSCAN(eps=0.09, min_samples=10)
                sample_labels = clusterer.fit_predict(sample_embeddings)
                
                # Calculate similarity to original clustering
                original_labels = self.labels[indices]
                stability = adjusted_rand_score(original_labels, sample_labels)
                stability_scores.append(max(0, stability))  # Ensure non-negative
            
            return np.mean(stability_scores)
            
        except Exception as e:
            logging.warning(f"Error calculating cluster stability: {e}")
            return 0.0
    
    def noise_ratio(self) -> float:
        noise_count = np.sum(self.labels == -1)
        return noise_count / len(self.labels)
    
    def cluster_size_distribution(self) -> Dict[str, float]:
        # Filter out noise points
        valid_labels = [label for label in self.unique_labels if label != -1]
        cluster_sizes = [len(self.label2docs[label]) for label in valid_labels]
        
        if not cluster_sizes:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "cv": 0}
        
        return {
            "mean": np.mean(cluster_sizes),
            "std": np.std(cluster_sizes),
            "min": np.min(cluster_sizes),
            "max": np.max(cluster_sizes),
            "cv": np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
        }
    
    def intra_cluster_distance(self) -> float:
        try:
            from sklearn.metrics.pairwise import pairwise_distances
            
            mask = self.labels != -1
            if np.sum(mask) < 2:
                return float('inf')
                
            filtered_embeddings = self.embeddings[mask]
            filtered_labels = self.labels[mask]
            valid_labels = np.unique(filtered_labels)
            
            if len(valid_labels) == 0:
                return float('inf')
            
            # Calculate all pairwise distances at once
            distances = pairwise_distances(filtered_embeddings)
            
            total_distance = 0
            total_pairs = 0
            
            for label in valid_labels:
                label_mask = filtered_labels == label
                label_indices = np.where(label_mask)[0]
                
                if len(label_indices) > 1:
                    # Get distances within this cluster
                    cluster_distances = distances[np.ix_(label_indices, label_indices)]
                    # Get upper triangle (excluding diagonal)
                    triu_indices = np.triu_indices_from(cluster_distances, k=1)
                    cluster_dist_values = cluster_distances[triu_indices]
                    total_distance += np.sum(cluster_dist_values)
                    total_pairs += len(cluster_dist_values)
            
            return total_distance / total_pairs if total_pairs > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"Error calculating intra-cluster distance: {e}")
            return float('inf')
    
    def inter_cluster_distance(self) -> float:
        try:
            from sklearn.metrics.pairwise import pairwise_distances
            
            mask = self.labels != -1
            if np.sum(mask) < 2:
                return 0.0
                
            filtered_embeddings = self.embeddings[mask]
            filtered_labels = self.labels[mask]
            valid_labels = np.unique(filtered_labels)
            
            if len(valid_labels) < 2:
                return 0.0
            
            # Calculate cluster centroids efficiently
            centroids = []
            for label in valid_labels:
                label_mask = filtered_labels == label
                centroid = np.mean(filtered_embeddings[label_mask], axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            # Calculate pairwise distances between centroids
            centroid_distances = pairwise_distances(centroids)
            
            # Get upper triangle (excluding diagonal)
            triu_indices = np.triu_indices_from(centroid_distances, k=1)
            distances = centroid_distances[triu_indices]
            
            return np.mean(distances) if len(distances) > 0 else 0.0
            
        except Exception as e:
            logging.warning(f"Error calculating inter-cluster distance: {e}")
            return 0.0
    
    def evaluate_all(self) -> Dict[str, Union[float, Dict]]:
        metrics = {
            "silhouette_score": self.silhouette_score_metric(),
            "calinski_harabasz_index": self.calinski_harabasz_index(),
            "davies_bouldin_score": self.davies_bouldin_score_metric(),
            "dunn_index": self.dunn_index(),
            "semantic_coherence": self.semantic_coherence(),
            "noise_ratio": self.noise_ratio(),
            "n_clusters": self.n_clusters,
            "cluster_size_distribution": self.cluster_size_distribution(),
            "intra_cluster_distance": self.intra_cluster_distance(),
            "inter_cluster_distance": self.inter_cluster_distance(),
        }
        
        # Add stability if computationally feasible
        if len(self.embeddings) < 10000:  # Only for smaller datasets
            metrics["cluster_stability"] = self.cluster_stability()
        
        return metrics
    
    def print_evaluation_report(self) -> None:
        """
        Print a comprehensive evaluation report.
        """
        metrics = self.evaluate_all()
        
        print("=" * 50)
        print("CLUSTER EVALUATION REPORT")
        print("=" * 50)
        print(f"Dataset size: {self.n_samples}")
        print(f"Number of clusters: {metrics['n_clusters']}")
        print(f"Noise ratio: {metrics['noise_ratio']:.3f}")
        print()
        
        print("QUALITY METRICS:")
        print("-" * 20)
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_index']:.3f}")
        print(f"Davies-Bouldin Score: {metrics['davies_bouldin_score']:.3f}")
        print(f"Dunn Index: {metrics['dunn_index']:.3f}")
        print(f"Semantic Coherence: {metrics['semantic_coherence']:.3f}")
        print()
        
        print("CLUSTER STRUCTURE:")
        print("-" * 20)
        size_dist = metrics['cluster_size_distribution']
        print(f"Average cluster size: {size_dist['mean']:.1f}")
        print(f"Cluster size std: {size_dist['std']:.1f}")
        print(f"Size coefficient of variation: {size_dist['cv']:.3f}")
        print(f"Intra-cluster distance: {metrics['intra_cluster_distance']:.3f}")
        print(f"Inter-cluster distance: {metrics['inter_cluster_distance']:.3f}")
        
        if 'cluster_stability' in metrics:
            print(f"Cluster stability: {metrics['cluster_stability']:.3f}")
        
        print("=" * 50)
    
    def get_cluster_quality_score(self) -> float:
        metrics = self.evaluate_all()
        
        # Normalize metrics to [0, 1] range
        silhouette_norm = (metrics['silhouette_score'] + 1) / 2  # [-1, 1] -> [0, 1]
        
        # Calinski-Harabasz: use log scaling and cap at reasonable value
        ch_norm = min(1.0, np.log(max(1, metrics['calinski_harabasz_index'])) / 10)
        
        # Davies-Bouldin: lower is better, normalize by inverting and capping
        db_norm = 1 / (1 + metrics['davies_bouldin_score'])  # Lower DB is better
        
        # Dunn index: cap at 2 for normalization
        dunn_norm = min(1.0, metrics['dunn_index'] / 2)
        
        # Semantic coherence: already in [-1, 1], normalize to [0, 1]
        semantic_norm = (metrics['semantic_coherence'] + 1) / 2
        
        # Noise ratio penalty (lower noise is better)
        noise_penalty = 1 - metrics['noise_ratio']
        
        # Weighted average of normalized metrics
        weights = [0.2, 0.15, 0.15, 0.15, 0.25, 0.10]  # Adjust weights as needed
        components = [silhouette_norm, ch_norm, db_norm, dunn_norm, semantic_norm, noise_penalty]
        
        composite_score = np.average(components, weights=weights)
        return float(composite_score)


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray, 
                       texts: Optional[List[str]] = None) -> Dict[str, Union[float, Dict]]:
    evaluator = ClusterEvaluator(embeddings, labels, texts)
    return evaluator.evaluate_all()


def quick_evaluation(embeddings: np.ndarray, labels: np.ndarray) -> float:
    evaluator = ClusterEvaluator(embeddings, labels)
    return evaluator.get_cluster_quality_score()
