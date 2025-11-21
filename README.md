# Movie Recommendation System

A reproducible project for building and evaluating movie recommendation models (collaborative, content-based, or hybrid). Includes data preparation, model training, offline evaluation, and example inference scripts.

## Features
- Training pipelines for collaborative and content-based models
- Evaluation with RMSE, Precision@K, Recall@K, NDCG
- Reproducible experiments via config files
- Example notebooks for exploratory analysis

## Project structure
- README.md — this file  
- data/ — raw and processed datasets (not checked into VCS)  
- notebooks/ — EDA and experiments (.ipynb)  
- src/ — core code (data loaders, models, training, evaluation)  
- scripts/ — runnable scripts: train.py, predict.py, evaluate.py  
- models/ — saved model checkpoints  
- requirements.txt — Python dependencies  
- config/ — example configs (YAML/JSON)

## Requirements
- Python 3.8+  
- Typical packages: numpy, pandas, scikit-learn, scipy, matplotlib, seaborn  
- Optional (deep learning): PyTorch or TensorFlow  
Install:
```
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
This project assumes MovieLens (e.g., 100k / 1M) or any user-item ratings file with columns: userId, movieId, rating, timestamp (optional). Place raw files under `data/raw/` and run the preprocessing script:
```
python src/data/preprocess.py --input data/raw/ratings.csv --output data/processed/
```

## Quick start
Train a model:
```
python scripts/train.py --config config/example.yaml
```
Evaluate:
```
python scripts/evaluate.py --model models/best_model.pth --test data/processed/test.csv
```
Generate recommendations:
```
python scripts/predict.py --model models/best_model.pth --user-id 123 --top-k 10 --output recommendations.csv
```

## Metrics
Commonly used metrics are:
- RMSE / MAE for rating prediction  
- Precision@K, Recall@K, NDCG@K, MAP for top-K recommendation

## Configuration
Use the config files under `config/` to control model type, hyperparameters, data paths, and training settings. Example fields: model.name, model.embedding_dim, training.batch_size, training.epochs, data.path.

## Experiments & Notebooks
Open `notebooks/` for exploratory analysis, baseline comparisons, and visualizations.

## Contributing
- Fork the repo, create a feature branch, open a pull request
- Run tests and linters before submitting
- Open issues for bugs or feature requests

## License
MIT License — see LICENSE file.

## Support
Open an issue in the repository for bugs, questions, or feature requests.
