
# Genetic Programming Symbolic Regression (RF + GP Hybrid)

This project implements a hybrid regression pipeline that combines:

- **Baseline models**: Linear Regression and Random Forest (for benchmarking).
- **Model-based feature selection**: Random Forest Gini importance to prune low-signal features.
- **Symbolic Regression via Genetic Programming**: `gplearn.SymbolicRegressor` with an extended function set (`+, -, *, /, sqrt, log, abs, neg, sin, cos`), tuned to work both on quasi-linear and highly non-linear datasets.

The system has been tested on:

- **Diabetes** dataset (biomedical, noisy, quasi-linear)
- **California Housing** dataset (spatial, non-linear)

The script trains baselines, selects features, evolves a symbolic model, and saves metrics, formulas, predictions, and comparison plots. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Repository Structure

Typical layout:

```text
.
├── genetic.py               # Main pipeline: RF + GP + reporting
├── requirements.txt         # Python package dependencies
├── raw_data/
│   ├── California.csv       # California Housing data (with "target" column)
│   └── Diabetes.csv         # Diabetes data (with "target" column)
└── processed_data/          # Auto-created: timestamped result folders
````

* `genetic.py` controls which dataset is used via the `RAW_DATA_FILE` constant. 
* `processed_data/` is created automatically per run using a timestamp and the seed. 

---

## Requirements

The project uses Python and standard ML/scientific libraries:

* Python **3.10+** (tested with modern Python; `requirements.txt` is compatible with 3.14+)
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `gplearn` (0.4.2+) 

Install everything with:

```bash
pip install -r requirements.txt
```

---

## Datasets

The script expects CSV files in `raw_data/` with:

* **California**:

  * Example filename: `raw_data/California.csv`
  * Features: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`
  * Target column: `target` (median house value) 

* **Diabetes**:

  * Example filename: `raw_data/Diabetes.csv`
  * Features: `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`
  * Target column: `target` (disease progression) 

You can generate these CSVs from `sklearn.datasets` beforehand, as you did for the report.

---

## Configuration

All main configuration is at the top of `genetic.py`:

```python
SEED: int = 42
RAW_DATA_FILE = "raw_data/California.csv"
RESULT_COLUMN = "target"
TEST_SIZE: float = 0.2
RESULT_DIR: str = "./processed_data/"
```

To switch datasets:

* **California Housing run** (default):

  * Ensure `RAW_DATA_FILE = "raw_data/California.csv"`.

* **Diabetes run**:

  * Change `RAW_DATA_FILE` to point to your Diabetes CSV, e.g.:

    ```python
    RAW_DATA_FILE = "raw_data/Diabetes.csv"
    ```

The rest of the pipeline (baselines, RF feature selection, GP configuration) is shared. 

---

## How to Run

From the root of the project (where `genetic.py` lives):

```bash
python genetic.py
```

This will:

1. Load the dataset defined in `RAW_DATA_FILE`.
2. Split into train/test (80/20, `seed=42`).
3. Train **Linear Regression** and **Random Forest** baselines and print their MAE scores.
4. Use Random Forest feature importance to select high-signal features.
5. Train a `SymbolicRegressor` (population 5000, 80 generations) on standardized inputs/targets.
6. Evaluate all models on the test set.
7. Save metrics, evolved formula, predictions, and plots into a timestamped folder inside `processed_data/`.  

If you want a log file similar to `EjecucionCalifornia.txt` / `EjecucionDiabetes.txt`, redirect stdout:

```bash
# California
python genetic.py > EjecucionCalifornia.txt

# After editing RAW_DATA_FILE to Diabetes.csv
python genetic.py > EjecucionDiabetes.txt
```

---

## Outputs

Each run creates a folder like:

```text
processed_data/2025.12.09_21.35.10_seed.42/
    mae_results.csv
    gplearn_model.json
    LinearRegression_predictions.csv
    gplearn_GP_predictions.csv
    predictions_comparison.png
```

* **`mae_results.csv`**: Train/test MAE for each model (LinearRegression, RandomForest, gplearn_GP). 
* **`gplearn_model.json`**: Evolved symbolic expression, hyperparameters, and metadata. 
* **`*_predictions.csv`**: For each model, the columns:

  * `y_prediction`
  * `y_actual` 
* **`predictions_comparison.png`**:

  * Scatter plots of actual vs predicted for each model, including the `y = x` reference line.  

The console also prints a **final ranking** of models by test MAE and the percentage improvement of GP over Linear Regression, as shown in the `Ejecucion*.txt` logs.  

---

## Reproducing the Report Results

To match the case studies in the report:

1. **Diabetes**:

   * Set `RAW_DATA_FILE` to the Diabetes CSV.
   * Run `python genetic.py`.
   * You should see:

     * Linear Regression test MAE ≈ 42.79
     * GP test MAE ≈ 40.21 (≈ 6% improvement vs Linear Regression) 

2. **California Housing**:

   * Set `RAW_DATA_FILE` to the California CSV.
   * Run `python genetic.py`.
   * You should see:

     * Linear Regression test MAE ≈ 0.5332
     * GP test MAE ≈ 0.4751 (≈ 10.9% improvement vs Linear Regression)  

The Random Forest baseline will generally be the strongest model on California and slightly overfit on Diabetes, consistent with the analysis in the report.  

---

## Notes & Troubleshooting

* The script includes a small compatibility patch so that `gplearn` works with newer versions of `scikit-learn` (it restores the internal `_validate_data` method expected by `gplearn`). 
* If you change the dataset:

  * Keep a `target` column name, or update `RESULT_COLUMN` in `genetic.py`.
  * Ensure all feature columns are numeric and convertible to `float64`. 
* For longer runs (especially California with 20k+ samples and a population of 5000), training can take several minutes; monitor progress via the generation table printed on screen. 

---
