ds_project/
│
├── README.md
├── requirements.txt
├── config/                  # YAML configs for params, paths, thresholds
│   └── config.yaml
│
├── tests/                   # Unit tests
│   └── test_pipeline.py
│
├── src/
│   ├── __init__.py
│   ├── pipeline/            # Main training & inference orchestrators
│   │   └── training_pipeline.py
│   │
│   ├── ingestion/           # data handale, connector, read and insert data
│   │   └── data_reader.py
│   │   └── api_reader.py
│   │
│   ├── tracker/           # mlflow utils, expriment tracker, model_deployment
│   │   └── expriment_tracker.py
│   │   └── model_deployment.py
│   │
│   ├── preprocessing/       # Data cleaning, feature engineering
│   │   └── transform.py
│   │   └── encoding.py
│   │
│   ├── prediction/          # Inference code (e.g., fastapi-compatible)
│   │   └── predict.py
│   │
│   ├── monitoring/          # Model drift, data validation, alerts, accuracy_metrics
│   │   └── data_drift.py
│   │   └── model_drift.py

