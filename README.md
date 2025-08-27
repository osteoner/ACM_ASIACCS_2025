# LLMHackScape

**Can We Unmask the Underground? Detecting and Predicting Hidden Forum Interactions**

A framework for community analysis in underground forums using LLM embeddings and clustering techniques.

## Installation

```bash
git clone https://github.com/XXXXX/LLMHackScape.git
cd LLMHackScape
pip install -r requirements.txt
```

## Required Dependencies

```bash
pip install torch transformers scikit-learn pandas numpy psycopg2-binary matplotlib seaborn plotly
```

## Project Structure

```
LLMHackScape/
├── src/
│   └── user_clustering.py       # ClusterClassifier class
├── cluster_evaluation.py        # ClusterEvaluator and evaluation functions
├── run_pipeline.py              # Main execution script
├── requirements.txt
└── README.md
```

## Core Libraries

### 1. ClusterClassifier (`src/user_clustering.py`)

Main clustering class that handles:
- LLM embedding generation
- HDBSCAN clustering
- Cluster summary generation

```python
from src.user_clustering import ClusterClassifier

cc = ClusterClassifier(embed_device=0)
embeddings, labels, summaries = cc.fit(texts)
```

### 2. Cluster Evaluation (`cluster_evaluation.py`)

Provides comprehensive clustering evaluation:

```python
from cluster_evaluation import ClusterEvaluator, evaluate_clustering, quick_evaluation

evaluator = ClusterEvaluator(embeddings, cluster_labels, texts)
evaluator.print_evaluation_report()

# Get composite quality score
quality_score = evaluator.get_cluster_quality_score()
```

## How to Run

### Option 1: Using run.py (Recommended)

The `run.py` script handles the complete workflow with command-line arguments:

```bash
# Basic usage with existing SUP data
python run.py --input_file user_representations.json --output_dir ./results

# With parameter optimization
python run.py --input_file data.json --optimize_params --output_dir ./results

# Generate SUP sequences from database first
python run.py --generate_sup --db_host localhost --db_name nulled --output_dir ./results
```

#### Key Arguments:

| Argument | Description | Example |
|----------|-------------|---------|
| `--input_file` | Path to JSON file with SUP sequences | `data.json` |
| `--output_dir` | Directory to save results | `./results` |
| `--n_samples` | Limit number of samples | `50000` |
| `--embed_device` | GPU device ID | `0` |
| `--dbscan_eps` | HDBSCAN epsilon parameter | `0.1` |
| `--dbscan_min_samples` | HDBSCAN min samples | `10` |
| `--optimize_params` | Enable parameter optimization | `--optimize_params` |

#### Database Parameters (for SUP generation):

| Argument | Description | Default |
|----------|-------------|---------|
| `--db_host` | Database host | `localhost` |
| `--db_name` | Database name | `nulled` |
| `--db_port` | Database port | `5434` |
| `--start_date` | Start date for data | `2018-01-01` |
| `--end_date` | End date for data | `2024-12-31` |
| `--min_threads` | Min threads per user | `4` |
| `--min_posts` | Min posts per user | `4` |

### Option 2: Step-by-Step Manual Process

#### Step 1: Generate SUP Sequences (if needed)

```python
import psycopg2
import json

# Connect to database
conn = psycopg2.connect(
    host='localhost', 
    dbname='nulled', 
    port=5434, 
    user='', 
    password=''
)

# Generate SUP sequences (you need to implement the generate_sup_json function)
sup_json = generate_sup_json(
    conn, 
    start_date='2018-01-01', 
    end_date='2024-12-31', 
    min_threads=4,
    min_posts=4,
    n_processes=7
)

# Save to file
with open("user_representations.json", "w") as f:
    json.dump(sup_json, f, indent=2)

conn.close()
```

#### Step 2: Load Data and Perform Clustering

```python
import pandas as pd
from src.user_clustering import ClusterClassifier
from cluster_evaluation import ClusterEvaluator

# Load data
df = pd.read_json('user_representations.json')
texts = df['sup_sequence'].tolist()

# Initialize classifier
cc = ClusterClassifier(embed_device=0)

# Fit clustering
embeddings, labels, summaries = cc.fit(texts)

# Evaluate results
evaluator = ClusterEvaluator(embeddings, labels, texts)
evaluator.print_evaluation_report()

quality_score = evaluator.get_cluster_quality_score()
print(f"Quality Score: {quality_score:.3f}")
```

#### Step 3: Parameter Optimization (Optional)

```python
from cluster_evaluation import quick_evaluation

def optimize_parameters(texts, param_grid):
    best_score = 0
    best_params = None
    
    for eps in param_grid['eps']:
        for min_samples in param_grid['min_samples']:
            print(f"Testing eps={eps}, min_samples={min_samples}")
            
            classifier = ClusterClassifier(
                dbscan_eps=eps,
                dbscan_min_samples=min_samples,
                summary_create=False
            )
            
            embeddings, labels, _ = classifier.fit(texts)
            score = quick_evaluation(embeddings, labels)
            
            if score > best_score:
                best_score = score
                best_params = {'eps': eps, 'min_samples': min_samples}
    
    return best_params, best_score

# Run optimization
param_grid = {
    'eps': [0.05, 0.09, 0.15, 0.2],
    'min_samples': [5, 10, 15, 20]
}

best_params, best_score = optimize_parameters(texts, param_grid)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.3f}")
```

## Usage Examples

### Basic Clustering

```bash
python run.py --input_file user_data.json --output_dir ./results --n_samples 10000
```

### With GPU Acceleration

```bash
python run.py --input_file user_data.json --embed_device 0 --device cuda --output_dir ./results
```

### Parameter Optimization

```bash
python run.py \
    --input_file user_data.json \
    --optimize_params \
    --eps_range 0.05 0.1 0.15 0.2 \
    --min_samples_range 5 10 15 20 \
    --output_dir ./optimized_results
```

### Generate SUP and Cluster

```bash
python run.py \
    --generate_sup \
    --db_host localhost \
    --db_name your_forum_db \
    --db_user username \
    --db_password password \
    --start_date 2018-01-01 \
    --end_date 2024-12-31 \
    --output_dir ./complete_analysis
```

## Output Files

The framework generates several output files:

```
results/
├── cluster_labels.json          # Cluster assignments for each user
├── cluster_summaries.json       # Generated summaries for each cluster
├── evaluation_metrics.json      # Clustering quality metrics 
└── optimization_results.json    # Parameter optimization results (if run)
```

## Data Format

### Input Format (SUP sequences)

```json
[
  {
    "user_id": "user_123",
    "sup_sequence": "malware_dev_2023-01 exploit_trade_2023-02 darknet_market_2023-03"
  },
  {
    "user_id": "user_456", 
    "sup_sequence": "cryptocurrency_fraud_2023-01 social_eng_2023-02"
  }
]
```

### Output Format (Cluster Labels)

```json
[0, 1, 0, 2, -1, 1, 0, 2, ...]
```

Where:
- `0, 1, 2, ...` are cluster IDs
- `-1` indicates noise/outlier points

## Evaluation Metrics

The framework provides several clustering quality metrics:

- **Silhouette Score**: Measures cluster cohesion and separation (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Average similarity between clusters (≥0, lower is better)  
- **Calinski-Harabasz Index**: Ratio of between/within cluster dispersion (higher is better)
- **Semantic Coherence**: Measures how semantically related the user sequences within each cluster are (0 to 1, higher is better)
- **Composite Quality Score**: Weighted combination of all metrics (0 to 1, higher is better)

## Configuration

### ClusterClassifier Parameters

```python
ClusterClassifier(
    embed_device=0,              # GPU device for embeddings
    dbscan_eps=0.1,             # HDBSCAN epsilon parameter
    dbscan_min_samples=10,      # HDBSCAN min samples parameter  
    summary_create=True         # Generate cluster summaries
)
```

### Common Parameter Ranges

- **eps**: 0.05 - 0.3 (smaller values = more clusters)
- **min_samples**: 3 - 50 (larger values = fewer, denser clusters)

## Performance Tips

1. **Use GPU**: Set `--embed_device 0` for faster embedding generation
2. **Limit Data Size**: Use `--n_samples` for large datasets during testing
3. **Parameter Optimization**: Start with broad ranges, then narrow down
4. **Memory Management**: Monitor memory usage with large datasets

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `n_samples` or use CPU (`--device cpu`)
2. **Database connection error**: Check database parameters and credentials
3. **Import errors**: Ensure all dependencies are installed
4. **Empty clusters**: Try different eps/min_samples values

### Debug Mode

Add verbose logging:

```bash
python run.py --input_file data.json --output_dir ./results --verbose
```

