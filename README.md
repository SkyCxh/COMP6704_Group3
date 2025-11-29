# Constrained Optimization for Large Language Models

This project implements various constrained optimization algorithms for classification problems in large language model embedding spaces. The focus is on optimization methods that maintain high accuracy while satisfying multiple constraints (such as L2 norm and box constraints).

## Project Overview

The project implements the following constrained optimization algorithms:

1. **Projected Gradient Descent (PGD)** - Full-batch projected gradient descent with projection steps to ensure constraint satisfaction
2. **Stochastic Projected Gradient Descent (SPGD)** - Mini-batch version of PGD suitable for large-scale datasets
3. **Penalty Method** - Transforms constraints into penalty terms added to the objective function
4. **Barrier Method** - Uses logarithmic barrier functions to handle constraints

All algorithms are applied to text classification tasks based on large language model embeddings, supporting both binary classification (logistic regression) and multi-class classification (softmax regression) problems.

## Setup

### Dependencies

```bash
pip install -r requirements.txt
```


### Data Preparation

The project supports multiple datasets, including IMDB (binary) and AG News (multi-class). First, extract text embeddings:

```bash
# Extract text embeddings for IMDB dataset
python -m scripts.extract_embeddings --config configs/imdb.yaml

# Extract text embeddings for AG News dataset
python -m scripts.extract_embeddings --config configs/ag_news.yaml
```

Extracted embeddings will be saved in the `data/processed/` directory.

## Running Experiments

Run all optimization algorithms and compare results:

```bash
# Run all algorithms on IMDB dataset
python -m src.train_eval --config configs/imdb.yaml

# Run all algorithms on AG News dataset
python -m src.train_eval --config configs/ag_news.yaml
```

## Configuration Files

Experiment parameters can be flexibly adjusted through YAML configuration files. Key configuration sections include:

- **dataset** - Dataset name and sample count limit
- **embeddings** - Text embedding model and batch size
- **optimization** - Optimization parameters (learning rate, iteration count, constraint parameters, etc.)
- **experiment** - Train/validation split ratio and random seed
- **paths** - Data storage paths

Example configurations can be found in `configs/imdb.yaml` and `configs/ag_news.yaml`.

## Project Structure

Please download the [dataset](https://drive.google.com/drive/folders/1hFptMiOU6QmqqqXJbZFG6aCzx9RB_-8n?usp=drive_link)

```
.
├── configs/                # Configuration files
│   ├── ag_news.yaml        # AG News dataset config
│   └── imdb.yaml           # IMDB dataset config
├── data/                   # Data directory
│   └── processed/          # Processed embedding data
│       ├── ag_news/        # AG News dataset embeddings
│       └── imdb/           # IMDB dataset embeddings
├── results/                # Experiment results
│   ├── figures/            # Chart outputs
│   └── metrics/            # Numerical metrics
├── scripts/                # Utility scripts
│   └── extract_embeddings.py  # Script to extract text embeddings
├── src/                    # Source code
│   ├── algorithms/         # Optimization algorithm implementations
│   │   ├── barrier.py      # Barrier method
│   │   ├── penalty.py      # Penalty method
│   │   ├── pgd.py          # Projected gradient descent
│   │   ├── solver_baseline.py  # SciPy solver baseline
│   │   └── spgd.py         # Stochastic projected gradient descent
│   ├── config.py           # Configuration management
│   ├── data_utils.py       # Data processing utilities
│   ├── embeddings.py       # Embedding model loading and processing
│   ├── model.py            # Model definitions (logistic regression, softmax regression)
│   ├── plots.py            # Visualization utilities
│   └── train_eval.py       # Training and evaluation main program
└── requirements.txt        # Project dependencies
```

## Experiment Results

Experiment results are saved in the `results/` directory:

- **figures/** - Contains convergence curves and final accuracy bar charts
- **metrics/** - Contains detailed performance metrics for each algorithm (JSON format)

## Constraints

This project considers two types of constraints:

1. **L2 Norm Constraint**: `||w||_2 <= R`, limiting the L2 norm of the weight vector
2. **Box Constraint**: `|w_j| <= C`, limiting the absolute value of each weight component

## Custom Experiments

To run custom experiments, you can:

1. Create a new configuration file (reference existing configurations)
2. Adjust optimization parameters (learning rate, constraint parameters, etc.)
3. Run experiments with the new configuration


## License

[MIT License](LICENSE)
