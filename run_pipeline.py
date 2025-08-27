#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import psycopg2
from pathlib import Path
import sys
import os

from src.user_clustering import ClusterClassifier
from cluster_evaluation import ClusterEvaluator, evaluate_clustering, quick_evaluation
from user_sequence import *


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Community Analysis with LLM Embeddings")
    
    # Data parameters
    parser.add_argument("--input_file", type=str, 
                       help="Path to input JSON file with SUP sequences", default="user_representations.json")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Directory to save results")
    parser.add_argument("--n_samples", type=int, default=100_000,
                       help="Maximum number of samples to process")
    parser.add_argument("--start", type=int, default=-1,
                       help="Start index for data slice")
    parser.add_argument("--end", type=int, default=100_000,
                       help="End index for data slice")
    
    # Device and processing parameters
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device for embeddings (cuda/cpu)")
    parser.add_argument("--embed_device", type=int, default=0,
                       help="GPU device ID for embeddings")
    
    # Database parameters (for SUP generation)
    parser.add_argument("--generate_sup", action="store_true",
                       help="Generate SUP sequences from database")
    parser.add_argument("--db_host", type=str, default="localhost",
                       help="Database host")
    parser.add_argument("--db_name", type=str, default="nulled",
                       help="Database name")
    parser.add_argument("--db_port", type=int, default=5434,
                       help="Database port")
    parser.add_argument("--db_user", type=str, default="",
                       help="Database user")
    parser.add_argument("--db_password", type=str, default="",
                       help="Database password")
    parser.add_argument("--start_date", type=str, default="2018-01-01",
                       help="Start date for SUP generation")
    parser.add_argument("--end_date", type=str, default="2024-12-31",
                       help="End date for SUP generation")
    parser.add_argument("--min_threads", type=int, default=4,
                       help="Minimum threads for user activity filter")
    parser.add_argument("--min_posts", type=int, default=4,
                       help="Minimum posts for user activity filter")
    parser.add_argument("--n_processes", type=int, default=7,
                       help="Number of processes for SUP generation")
    
    # Clustering parameters
    parser.add_argument("--dbscan_eps", type=float, default=0.1,
                       help="DBSCAN eps parameter")
    parser.add_argument("--dbscan_min_samples", type=int, default=10,
                       help="DBSCAN min_samples parameter")
    parser.add_argument("--summary_create", action="store_true",
                       help="Create cluster summaries")
    
    # Evaluation and optimization
    parser.add_argument("--optimize_params", action="store_true",
                       help="Perform parameter optimization")
    parser.add_argument("--eps_range", type=float, nargs='+', 
                       default=[0.05, 0.09, 0.15, 0.2],
                       help="List of eps values to test during optimization")
    parser.add_argument("--min_samples_range", type=int, nargs='+',
                       default=[5, 10, 15, 20],
                       help="List of min_samples values to test during optimization")
    
    return parser.parse_args()


def generate_sup_sequences(args):
    """Generate SUP sequences from database."""
    print("Connecting to database...")
    conn = psycopg2.connect(
        host=args.db_host, 
        dbname=args.db_name, 
        port=args.db_port, 
        user=args.db_user, 
        password=args.db_password
    )
    
    print("Generating SUP sequences...")
    # Note: This function needs to be imported from your module
    # from your_module import generate_sup_json
    sup_json = generate_sup_json(
        conn, 
        start_date=args.start_date, 
        end_date=args.end_date, 
        min_threads=args.min_threads,
        min_posts=args.min_posts,
        n_processes=args.n_processes
    )
    
    # Save to file
    output_file = Path(args.output_dir) / f"user_representations_{args.db_name}_{args.start_date.replace('-', '')}_{args.end_date.replace('-', '')}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(sup_json, f, indent=2)
    
    print(f"Saved {len(sup_json)} SUP sequences to {output_file}")
    conn.close()
    
    return str(output_file)


def load_data(input_file, args):
    """Load and preprocess data."""
    print(f"Loading data from {input_file}...")
    
    if input_file.endswith('.json'):
        df = pd.read_json(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")
    
    # Apply data slicing if specified
    if args.start >= 0 or args.end < len(df):
        start_idx = max(0, args.start) if args.start >= 0 else 0
        end_idx = min(len(df), args.end)
        df = df.iloc[start_idx:end_idx]
        print(f"Using data slice [{start_idx}:{end_idx}], {len(df)} samples")
    
    # Limit samples if specified
    if len(df) > args.n_samples:
        df = df.head(args.n_samples)
        print(f"Limited to {args.n_samples} samples")
    
    return df


def cluster_fit_with_evaluation(classifier, texts, embeddings=None):
    """Perform clustering with comprehensive evaluation."""
    print("Fitting clustering model...")
    
    # Original clustering
    embeddings, cluster_labels, cluster_summaries = classifier.fit(texts, embeddings)
    
    # Evaluate clustering quality
    print("\nEvaluating clustering quality...")
    evaluator = ClusterEvaluator(embeddings, cluster_labels, texts)
    
    # Print comprehensive report
    evaluator.print_evaluation_report()
    
    # Get all metrics
    metrics = evaluator.evaluate_all()
    
    # Get composite quality score
    quality_score = evaluator.get_cluster_quality_score()
    print(f"\nComposite Quality Score: {quality_score:.3f}")
    
    return embeddings, cluster_labels, cluster_summaries, metrics, quality_score


def parameter_optimization(texts, args):
    """Perform parameter optimization for clustering."""
    print("\nStarting parameter optimization...")
    
    param_grid = {
        'eps': args.eps_range,
        'min_samples': args.min_samples_range
    }
    
    best_score = 0
    best_params = None
    best_results = None
    
    total_combinations = len(param_grid['eps']) * len(param_grid['min_samples'])
    current_combination = 0
    
    for eps in param_grid['eps']:
        for min_samples in param_grid['min_samples']:
            current_combination += 1
            print(f"\n[{current_combination}/{total_combinations}] Testing eps={eps}, min_samples={min_samples}")
            
            # Create classifier with current parameters
            classifier = ClusterClassifier(
                embed_device=args.embed_device,
                dbscan_eps=eps,
                dbscan_min_samples=min_samples,
                summary_create=False  # Skip summaries for speed during optimization
            )
            
            # Fit and evaluate
            embeddings, labels, summaries = classifier.fit(texts)
            
            # Quick evaluation
            score = quick_evaluation(embeddings, labels)
            print(f"Quality score: {score:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
                best_results = (embeddings, labels, summaries, classifier)
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best quality score: {best_score:.3f}")
    
    return best_results, best_params, best_score


def save_results(results, output_dir, args):
    """Save clustering results and metrics."""
    embeddings, labels, summaries, metrics, quality_score = results
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cluster labels
    labels_file = output_dir / "cluster_labels.json"
    with open(labels_file, 'w') as f:
        json.dump(labels.tolist(), f)
    
    # Save cluster summaries if available
    if summaries:
        summaries_file = output_dir / "cluster_summaries.json"
        with open(summaries_file, 'w') as f:
            json.dump(summaries, f, indent=2)
    
    # Save metrics
    metrics_file = output_dir / "evaluation_metrics.json"
    metrics['quality_score'] = quality_score
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save embeddings (optional, can be large)
    # embeddings_file = output_dir / "embeddings.npy"
    # np.save(embeddings_file, embeddings)
    
    print(f"Results saved to {output_dir}")


def main():
    """Main execution function."""
    args = get_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate SUP sequences if requested
    if args.generate_sup:
        input_file = generate_sup_sequences(args)
        if not args.input_file:
            args.input_file = input_file
    
    # Check if input file exists
    if not args.input_file or not Path(args.input_file).exists():
        print("Error: No input file specified or file does not exist.")
        print("Use --input_file to specify input file or --generate_sup to generate from database.")
        sys.exit(1)
    
    # Load data
    df = load_data(args.input_file, args)
    texts = df['sup_sequence'].tolist()
    print(f"Loaded {len(texts)} text sequences")
    
    # Initialize classifier
    classifier = ClusterClassifier(
        embed_device=args.embed_device,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        summary_create=args.summary_create
    )
    
    if args.optimize_params:
        # Parameter optimization
        best_results, best_params, best_score = parameter_optimization(texts, args)
        embeddings, labels, summaries, best_classifier = best_results
        
        # Perform detailed evaluation with best parameters
        print(f"\nPerforming detailed evaluation with best parameters: {best_params}")
        embeddings, labels, summaries, metrics, quality_score = cluster_fit_with_evaluation(
            best_classifier, texts
        )
        
        # Save optimization results
        optimization_results = {
            'best_params': best_params,
            'best_score': best_score,
            'optimization_args': {
                'eps_range': args.eps_range,
                'min_samples_range': args.min_samples_range
            }
        }
        
        opt_file = Path(args.output_dir) / "optimization_results.json"
        with open(opt_file, 'w') as f:
            json.dump(optimization_results, f, indent=2)
        
    else:
        # Standard clustering with evaluation
        embeddings, labels, summaries, metrics, quality_score = cluster_fit_with_evaluation(
            classifier, texts
        )
    
    # Save results
    save_results((embeddings, labels, summaries, metrics, quality_score), args.output_dir, args)
    
    # Quick final evaluation
    if hasattr(classifier, 'embeddings') and hasattr(classifier, 'cluster_labels'):
        final_score = quick_evaluation(classifier.embeddings, classifier.cluster_labels)
        print(f"\nFinal quick quality score: {final_score:.3f}")
    
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    main()