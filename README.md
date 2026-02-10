# Museums & Population — Data Pipeline Project

## Overview

This project is a small end-to-end **data pipeline and modeling exercise** built to demonstrate how raw public data can be collected, cleaned, merged, analyzed and used to train a simple model.

The main idea is to study the relationship between:
- the number of museums in a city
- the population of that city
- and derived metrics used for analysis and visualization

The project follows a **clear, reproducible sequence of steps**, from data collection to visualization.

---

## Project Goals

- Practice building a structured data pipeline in Python
- Work with real-world, imperfect data
- Separate exploration logic from production-ready code
- Prepare a clean dataset for modeling
- Train a simple model and visualize results

---

## Project Structure

```

project/
│
├── data/
│   ├── raw/                # Raw fetched data
│   └── processed/          # Cleaned and merged datasets
│
├── outputs/
│   ├── models/             # Trained models
│   └── figures/            # Generated plots
│
├── src/
│   ├── fetch_museums.py
│   ├── load_population.py
│   ├── join_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── visualize.py
│
├── notebooks/              # Exploration and experiments (optional)
└── README.md

````

---

## Pipeline Overview

The project is designed as a **linear pipeline**, where each step produces an output that is reused by the next step.

### 1. Fetch museums data

```bash
python -m src.fetch_museums
````

This step:

* Fetches museum-related data from Wikipedia
* Extracts city-level information
* Stores the result as a raw dataset

Output:

```
data/raw/museums.csv
```

---

### 2. Load population data

```bash
python -m src.load_population
```

This step:

* Loads population data from a structured source
* Normalizes city names
* Prepares a clean population table

Output:

```
data/raw/population.csv
```

---

### 3. Join museums and population data

```bash
python -m src.join_data --verbose
```

This step:

* Joins the museums dataset with population data
* Matches cities across both datasets
* Logs unmatched or ambiguous cases when `--verbose` is enabled

Output:

```
data/processed/museums_joined.csv
```

---

### 4. Preprocess the dataset

```bash
python -m src.preprocess \
  --in data/processed/museums_joined.csv \
  --out data/processed/dataset.csv
```

This step:

* Cleans missing or invalid values
* Creates derived features
* Produces the final dataset used for modeling

Output:

```
data/processed/dataset.csv
```

---

### 5. Train the model

```bash
python -m src.train_model \
  --in data/processed/dataset.csv \
  --out outputs/models
```

This step:

* Loads the processed dataset
* Trains a simple model
* Saves the trained model to disk

Output:

```
outputs/models/
```

---

### 6. Visualization and observations

```bash
python -m src.visualize \
  --in data/processed/dataset.csv \
  --out outputs/figures
```

This step:

* Generates plots and visualizations
* Helps analyze relationships in the data
* Saves figures for reporting and interpretation

Output:

```
outputs/figures/
```

---

## Design Philosophy

* **Notebooks are optional** and used only for exploration
* **All core logic lives in `.py` files**
* Each script can be executed independently
* Clear input/output paths for reproducibility
* Ready to be automated or integrated into a larger pipeline

---

## How to Run the Full Pipeline

From the project root:

```bash
python -m src.fetch_museums
python -m src.load_population
python -m src.join_data --verbose
python -m src.preprocess --in data/processed/museums_joined.csv --out data/processed/dataset.csv
python -m src.train_model --in data/processed/dataset.csv --out outputs/models
python -m src.visualize --in data/processed/dataset.csv --out outputs/figures
```

---

## Possible Extensions

* Add automated tests for each step
* Introduce logging and metrics
* Try alternative models
* Add confidence intervals or residual analysis
* Automate the pipeline using Airflow or similar tools

---

## Final Notes

This project is intentionally simple but structured in a way that mirrors real-world data workflows.
The focus is on **clarity, separation of concerns, and reproducibility**, not on model complexity.
