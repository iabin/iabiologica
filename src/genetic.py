"""
Genetic Programming Symbolic Regression
========================================

A robust Python implementation using `gplearn` for Symbolic Regression, benchmarked 
against standard `sklearn` models (LinearRegression, RandomForest).

System Architecture:
--------------------
1. Data Ingestion: Loads raw CSV data and ensures consistent float64 typing.
2. Baseline Benchmarking: Trains Linear Regression and Random Forest models to establish
   performance targets (Simplicity vs. Complexity).
3. Intelligent Feature Selection: Uses the Random Forest's feature importance scores to
   prune the input space for the Genetic Program. This helps the GP focus on signal rather 
   than noise.
4. Genetic Evolution: Trains a Symbolic Regressor using the "Universal V2" configuration,
   designed to handle both linear datasets (like Diabetes) and spatial/complex datasets
   (like California Housing) without manual tuning.
5. Reporting: Exports comparison metrics (MAE), the evolved mathematical formula, 
   raw predictions, and visual scatter plots.

Key Dependencies:
-----------------
- gplearn: For the evolutionary algorithm.
- scikit-learn: For baseline models, preprocessing, and metrics.
- pandas/numpy: For data manipulation.
- matplotlib: For visualization.

Usage:
------
    python genetic.py
"""

import pandas as pd
import numpy as np
import os
import json
import time
import warnings
from datetime import datetime

# Standard Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

# === GPLEARN COMPATIBILITY PATCH ===
# gplearn (as of v0.4.2) relies on an internal sklearn attribute `_validate_data`
# that was removed/refactored in scikit-learn 1.6+.
# This monkey-patch restores compatibility by mapping the old method call to the new validator.
from sklearn.utils.validation import validate_data
from gplearn import genetic as gplearn_genetic
gplearn_genetic.BaseSymbolic._validate_data = validate_data

from gplearn.genetic import SymbolicRegressor

# === GLOBAL CONFIGURATION ===
# SEED: Controls stochastic behavior for reproducibility across runs.
# RAW_DATA_FILE: Path to the dataset (switch between 'Diabetes.csv' or 'California.csv').
SEED: int = 42
RAW_DATA_FILE = "raw_data/California.csv"
RESULT_COLUMN = "target"
TEST_SIZE: float = 0.2
RESULT_DIR: str = "./processed_data/"


# === GPLEARN CONFIGURATION (UNIVERSAL V2) ===
# This configuration is tuned to be a "Swiss Army Knife" solver.
# It balances the need for simplicity (Linear Regression) with the capacity for
# complexity (Spatial Mapping using Sine/Cosine).

# POPULATION: 5000 is large. This ensures high diversity in the initial generation,
# increasing the probability of finding "rare" features like sin(Latitude) early on.
GP_POPULATION_SIZE = 5000 

# GENERATIONS: 80 gives the algorithm enough time to refine constants (point mutation)
# without running for so long that it memorizes noise (bloat).
GP_GENERATIONS = 80

# TOURNAMENT SIZE: 20 creates moderate evolutionary pressure.
# Higher numbers favor the "fittest" more aggressively; lower numbers preserve diversity.
GP_TOURNAMENT_SIZE = 20
GP_STOPPING_CRITERIA = 0.001
GP_CONST_RANGE = (-10.0, 10.0)
GP_INIT_DEPTH = (2, 6)

# THE GOLDILOCKS PARSIMONY COEFFICIENT (0.001)
# This is the "bloat control" penalty.
# - If too high (>0.0015): The model becomes "starved" and refuses to add useful terms (like BMI).
# - If too low (<0.0005): The model "bloats" with useless terms to chase tiny error reductions.
# - 0.001 is the sweet spot: It allows complexity ONLY if it significantly reduces error.
GP_PARSIMONY_COEF = 0.001

# THE STABLE FUNCTION SET
# - Arithmetic: Standard building blocks (add, sub, mul, div).
# - Transformations: log/sqrt/abs/neg to handle scaling and diminishing returns.
# - Trigonometry: sin/cos are CRITICAL for spatial datasets (creating "heatmaps" from Lat/Long).
# - NOTE: 'tan' is excluded because its asymptotes cause explosive error spikes.
GP_FUNCTION_SET = ("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "sin", "cos")

# EVOLUTION PROBABILITIES
# - Crossover (0.6): Main driver of evolution. Mixes parts of good formulas.
# - Point Mutation (0.25): High setting. Critical for "fine-tuning" the float constants
#   (e.g., turning 6.0 into 6.207) to beat Linear Regression on precision.
GP_P_CROSSOVER = 0.6 
GP_P_SUBTREE_MUTATION = 0.1
GP_P_HOIST_MUTATION = 0.05
GP_P_POINT_MUTATION = 0.25 
GP_MAX_SAMPLES = 0.95 # Use 95% of data per generation to prevent overfitting to a small subset.
GP_METRIC = "mean absolute error"

# FEATURE SELECTION PARAMETERS
# We use Random Forest importance to filter features before they reach the GP.
# This prevents the GP from wasting cycles evolving formulas with useless variables.
GP_MIN_FEATURES = 3        # Safety floor: always keep at least 3 features.
GP_IMPORTANCE_THRESHOLD = 0.04 # Keep features with >4% importance.


# === BASELINE MODELS ===
# We wrap models in a Pipeline to ensure robust handling of missing values (Imputer)
# and scaling (StandardScaler), ensuring fair comparison with the GP.
models = {
    "LinearRegression": make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()
    ),
    "RandomForest": make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1
        ),
    ),
}


# === DATA UTILITIES ===

def load_csv_as_dataframe(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file and enforces float64 data types for numerical columns.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: The loaded data with sanitized types.
    """
    df = pd.read_csv(filepath)
    # Ensure all numeric columns are strictly float64 to avoid type errors in sklearn/gplearn
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].to_numpy(dtype=np.float64)
    return df


def split_dataframe(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataframe into training and testing sets.
    
    Args:
        df (pd.DataFrame): The full dataset.
        test_size (float): Fraction of data to reserve for testing (0.0 to 1.0).
        seed (int): Random seed for reproducibility.
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, shuffle=True
    )
    # Reset indices to avoid misalignment errors during future concatenation/joins
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return (train_df, test_df)


def split_features_and_target(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separates the input features (X) from the target variable (y).
    
    Args:
        df (pd.DataFrame): The dataset.
        target_column (str): The name of the column to predict.
        
    Returns:
        tuple: (X dataframe, y series)
    """
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return (x, y)


# === RESULT MANAGEMENT UTILITIES ===

def create_result_dir(base_dir: str, seed: int) -> str:
    """
    Creates a unique, timestamped directory to store the results of this run.
    This prevents overwriting previous experiments.
    
    Args:
        base_dir (str): Root directory for processing results.
        seed (int): The seed used, appended to the directory name for tracking.
        
    Returns:
        str: The full path to the newly created directory.
    """
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    dir_name = f"{timestamp}_seed.{seed}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_dataframe(results: pd.DataFrame, output_dir: str, filename: str) -> None:
    """
    Helper to save a pandas DataFrame to CSV within the results directory.
    """
    results.to_csv(os.path.join(output_dir, filename), index=False)
    print(f"  {filename} saved to {output_dir}")


def plot_predictions_comparison(predictions: dict, output_dir: str) -> None:
    """
    Generates scatter plots comparing Predicted vs. Actual values for all models.
    
    - X-axis: Actual Values
    - Y-axis: Predicted Values
    - Red Dashed Line: Perfect prediction (x=y)
    
    This visualizes how well models fit the data and identifies bias/variance issues.
    
    Args:
        predictions (dict): Dictionary mapping model names to (y_pred, y_actual) tuples.
        output_dir (str): Directory to save the PNG image.
    """
    n_models = len(predictions)
    cols = 2
    rows = (n_models + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(12, 5 * rows))

    # Ensure axes is always iterable even if only 1 model
    axes = axes.flatten() if n_models > 1 else [axes]

    for idx, (name, (y_pred, y_test)) in enumerate(predictions.items()):
        ax = axes[idx]
        mae = mean_absolute_error(y_test, y_pred)
        ax.scatter(y_test, y_pred, alpha=0.5, s=10)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        # Plot the "Perfect Prediction" reference line
        ax.plot(
            [min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction"
        )
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{name} (MAE: {mae:.4f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide empty subplots if the grid is larger than the number of models
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "predictions_comparison.png"), dpi=150)
    plt.close()
    print(f"  Plot saved to {output_dir}/predictions_comparison.png")


# === GENETIC PROGRAMMING CORE FUNCTIONS ===

def train_gplearn_model(
    X_train, y_train, random_state: int = SEED
) -> SymbolicRegressor:
    """
    Initializes and trains the SymbolicRegressor.
    
    This function instantiates the GP model with the global configuration parameters
    defined at the top of the script and fits it to the scaled training data.
    
    Args:
        X_train (array-like): Scaled training features.
        y_train (array-like): Scaled training targets.
        random_state (int): Seed for reproducibility.

    Returns:
        SymbolicRegressor: The trained model object.
    """
    print(f"\n  Training gplearn SymbolicRegressor...")
    print(f"  Population: {GP_POPULATION_SIZE}, Generations: {GP_GENERATIONS}")
    print(f"  Tournament size: {GP_TOURNAMENT_SIZE}, Max samples: {GP_MAX_SAMPLES}")
    print(f"  Function set: {GP_FUNCTION_SET}")

    gp_model = SymbolicRegressor(
        population_size=GP_POPULATION_SIZE,
        generations=GP_GENERATIONS,
        tournament_size=GP_TOURNAMENT_SIZE,
        stopping_criteria=GP_STOPPING_CRITERIA,
        const_range=GP_CONST_RANGE,
        init_depth=GP_INIT_DEPTH,
        init_method="half and half",
        function_set=GP_FUNCTION_SET,
        metric=GP_METRIC,
        parsimony_coefficient=GP_PARSIMONY_COEF,
        p_crossover=GP_P_CROSSOVER,
        p_subtree_mutation=GP_P_SUBTREE_MUTATION,
        p_hoist_mutation=GP_P_HOIST_MUTATION,
        p_point_mutation=GP_P_POINT_MUTATION,
        max_samples=GP_MAX_SAMPLES,
        verbose=1, # Prints progress per generation
        n_jobs=-1, # Uses all available CPU cores
        random_state=random_state,
        warm_start=False,
        low_memory=True,
    )

    start_time = time.time()
    # Suppress warnings that might arise from internal gplearn operations during fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp_model.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"\n  gplearn training completed in {elapsed:.2f}s")
    print(f"  Best program: {gp_model._program}")
    print(f"  Best fitness: {gp_model._program.raw_fitness_:.4f}")

    return gp_model


def gplearn_to_expression_string(gp_model, feature_names: list) -> str:
    """
    Converts the evolved program (which uses generic placeholders like X0, X1)
    into a human-readable mathematical string using real feature names.
    
    Args:
        gp_model: The trained GP model.
        feature_names (list): List of strings corresponding to column names.
        
    Returns:
        str: The mathematical expression (e.g., "add(bmi, mul(s5, 0.5))").
    """
    # Access the program string. _program is the best individual from the final generation.
    if hasattr(gp_model, "_program"):
        expr = str(gp_model._program)
    else:
        expr = str(gp_model)
    
    # Replace X0, X1, etc., with actual names (e.g., 'bmi', 's5')
    for i, name in enumerate(feature_names):
        expr = expr.replace(f"X{i}", name)
    return expr


def select_features_for_gp(
    rf_pipeline,
    feature_names: list,
    importance_threshold: float = GP_IMPORTANCE_THRESHOLD,
    min_features: int = GP_MIN_FEATURES,
) -> list:
    """
    Performs feature selection to reduce the search space for Genetic Programming.
    
    It uses the Feature Importance attribute from a pre-trained Random Forest.
    Logic:
    1. Filter features that contribute > importance_threshold (e.g., 4%).
    2. Fallback: If too few features pass the threshold, keep the top `min_features` 
       to prevent starving the model of data.

    Args:
        rf_pipeline: Trained Pipeline containing a 'randomforestregressor'.
        feature_names (list): All available feature names.
        importance_threshold (float): Cutoff for feature inclusion.
        min_features (int): Minimum number of features to ensure are selected.

    Returns:
        list: The names of the features to use for GP training.
    """
    rf = rf_pipeline.named_steps["randomforestregressor"]
    importances = rf.feature_importances_
    
    # Create a sorted dataframe of feature importances
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    print("\n  Feature importances (RandomForest):")
    for _, row in importance_df.iterrows():
        pct = row["importance"] * 100
        bar = "█" * int(pct * 2)  # ASCII bar chart
        marker = "✓" if row["importance"] >= importance_threshold else " "
        print(f"  {marker} {row['feature']:20s} {pct:5.2f}% {bar}")

    # Strategy 1: Keep features meeting the threshold
    selected = importance_df[importance_df["importance"] >= importance_threshold][
        "feature"
    ].tolist()

    # Strategy 2 (Safety Net): If Strategy 1 removes too much, take the top N features
    if len(selected) < min_features:
        print(
            f"\n  Only {len(selected)} features above {importance_threshold*100:.1f}% threshold."
        )
        print(f"  Falling back to top-{min_features} features.")
        selected = importance_df["feature"].iloc[:min_features].tolist()

    print(f"\n  Selected {len(selected)} features for GP: {selected}")

    return selected


# === MAIN EXECUTION FLOW ===
if __name__ == "__main__":
    print("=" * 60)
    print("GENETIC PROGRAMMING SYMBOLIC REGRESSION")
    print("=" * 60)

    # 1. LOAD DATA
    print("\n[1] Loading raw data...")
    df = load_csv_as_dataframe(RAW_DATA_FILE)
    print(f"  Total samples: {len(df)}")
    print(f"  Features: {list(df.columns)}")

    # 2. SPLIT DATA
    print(f"\n[2] Splitting data (80/20, seed={SEED})...")
    train_df, test_df = split_dataframe(df, TEST_SIZE, SEED)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Separating X (features) and y (targets)
    x_train, y_train = split_features_and_target(train_df, RESULT_COLUMN)
    x_test, y_test = split_features_and_target(test_df, RESULT_COLUMN)
    feature_names = list(x_train.columns)

    # 3. BASELINE TRAINING (Benchmark phase)
    print("\n[3] Training sklearn baseline models...")
    trained_models = {}
    train_mae_results = {}
    test_mae_results = {}
    
    # Loop through baseline models (Linear Reg, Random Forest), train, and eval
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(x_train, y_train)
        trained_models[name] = model

        # Evaluate on train
        y_pred_train = model.predict(x_train)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        train_mae_results[name] = mae_train

        # Evaluate on test
        y_pred_test = model.predict(x_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        test_mae_results[name] = mae_test

        print(f"  {name} - Train MAE: {mae_train:.4f}, Test MAE: {mae_test:.4f}")

    # 3.1. FEATURE SELECTION (Pruning phase)
    print("\n[3.1] Selecting features for GP (threshold-based):")
    print(
        f"  Config: importance_threshold={GP_IMPORTANCE_THRESHOLD*100:.1f}%, min_features={GP_MIN_FEATURES}"
    )
    # Use the pre-trained Random Forest to identify the most predictive features
    gp_feature_names = select_features_for_gp(
        rf_pipeline=trained_models["RandomForest"],
        feature_names=feature_names,
        importance_threshold=GP_IMPORTANCE_THRESHOLD,
        min_features=GP_MIN_FEATURES,
    )

    # Subset the data to include only the selected features for GP
    x_train_gp = x_train[gp_feature_names]
    x_test_gp = x_test[gp_feature_names]

    # 4. GP TRAINING (Evolution phase)
    print("\n[4] Training gplearn Symbolic Regression...")

    # Step 4a: Scale Inputs (X)
    # GP learns faster and more stably on standardized data (mean=0, var=1).
    gp_scaler_x = StandardScaler()
    x_train_scaled = gp_scaler_x.fit_transform(x_train_gp.values)
    x_test_scaled = gp_scaler_x.transform(x_test_gp.values)

    # Step 4b: Scale Targets (y)
    # CRITICAL: GP struggles to output raw values if they are large (e.g., house prices).
    # We scale targets to N(0,1), let GP predict in that space, then inverse transform back.
    gp_scaler_y = StandardScaler()
    y_train_scaled = gp_scaler_y.fit_transform(np.array(y_train).reshape(-1, 1)).ravel()

    # Step 4c: Train the model
    gplearn_model = train_gplearn_model(
        x_train_scaled, y_train_scaled, random_state=SEED
    )

    # Step 4d: Extract the formula
    gplearn_expr = gplearn_to_expression_string(gplearn_model, gp_feature_names)
    print(f"\n  gplearn evolved expression: {gplearn_expr}")

    # Step 4e: Generate Predictions and Descale
    # We must predict in the scaled space, then invert the y-scaler to get real-world units.
    
    # Train set predictions
    y_pred_scaled_train = gplearn_model.predict(x_train_scaled)
    y_pred_gplearn_train = gp_scaler_y.inverse_transform(
        y_pred_scaled_train.reshape(-1, 1)
    ).ravel()
    mae_gplearn_train = mean_absolute_error(y_train, y_pred_gplearn_train)
    train_mae_results["gplearn_GP"] = mae_gplearn_train

    # Test set predictions
    y_pred_scaled_test = gplearn_model.predict(x_test_scaled)
    y_pred_gplearn_test = gp_scaler_y.inverse_transform(
        y_pred_scaled_test.reshape(-1, 1)
    ).ravel()
    mae_gplearn_test = mean_absolute_error(y_test, y_pred_gplearn_test)
    test_mae_results["gplearn_GP"] = mae_gplearn_test

    print(
        f"\n  gplearn_GP - Train MAE: {mae_gplearn_train:.4f}, Test MAE: {mae_gplearn_test:.4f}"
    )

    # 5. RESULTS AGGREGATION
    test_predictions = {}

    # Gather LinearRegression predictions for comparison plotting
    if "LinearRegression" in trained_models:
        model = trained_models["LinearRegression"]
        y_pred = model.predict(x_test)
        test_predictions["LinearRegression"] = (
            pd.Series(y_pred),
            pd.Series(y_test.values),
        )

    # Gather GP predictions
    test_predictions["gplearn_GP"] = (
        pd.Series(y_pred_gplearn_test),
        pd.Series(y_test.values),
    )

    # 6. SAVE ARTIFACTS
    print("\n[5] Saving results...")
    result_dir = create_result_dir(RESULT_DIR, SEED)

    # 6a. MAE Results CSV
    mae_dataframe = pd.DataFrame(
        [
            {
                "Model": name,
                "Train_MAE": train_mae_results[name],
                "Test_MAE": test_mae_results[name],
            }
            for name in test_mae_results.keys()
        ]
    )
    save_dataframe(mae_dataframe, result_dir, "mae_results.csv")

    # 6b. Model Metadata (JSON)
    # Stores configuration and the evolved formula for reproducibility.
    function_set_names = list(GP_FUNCTION_SET)
    gplearn_info = {
        "expression": gplearn_expr,
        "train_mae": float(mae_gplearn_train),
        "test_mae": float(mae_gplearn_test),
        "population_size": GP_POPULATION_SIZE,
        "generations": GP_GENERATIONS,
        "function_set": function_set_names,
        "features_used": gp_feature_names,
        "total_features": len(feature_names),
        "feature_selection": {
            "method": "rf_importance_threshold",
            "importance_threshold": GP_IMPORTANCE_THRESHOLD,
            "min_features": GP_MIN_FEATURES,
        },
        "metric": GP_METRIC,
        "seed": SEED,
    }

    with open(os.path.join(result_dir, "gplearn_model.json"), "w") as f:
        json.dump(gplearn_info, f, indent=2)
    print(f"  gplearn_model.json saved to {result_dir}")

    # 6c. Raw Predictions CSVs
    for model_name, (y_pred, y_true) in test_predictions.items():
        pred_dataframe = pd.DataFrame(
            {"y_prediction": y_pred.values, "y_actual": y_true.values}
        )
        save_dataframe(pred_dataframe, result_dir, f"{model_name}_predictions.csv")

    # 7. VISUALIZATION
    print("\n[6] Creating comparison plot...")
    plot_predictions_comparison(test_predictions, result_dir)

    # 8. FINAL REPORT
    print("\n" + "=" * 60)
    print("FINAL RESULTS (TEST SET EVALUATION)")
    print("=" * 60)

    # Sort models by performance (Test MAE ascending)
    sorted_results = sorted(test_mae_results.items(), key=lambda x: x[1])

    for rank, (model_name, mae) in enumerate(sorted_results, 1):
        marker = "🏆" if rank == 1 else "  "
        print(f"  {marker} {rank}. {model_name}: Test MAE = {mae:.4f}")

    # Specific comparison: GP vs Linear Regression
    if "LinearRegression" in test_mae_results and "gplearn_GP" in test_mae_results:
        lr_mae = test_mae_results["LinearRegression"]
        gp_mae = test_mae_results["gplearn_GP"]
        improvement = (lr_mae - gp_mae) / lr_mae * 100
        if gp_mae < lr_mae:
            print(f"\n  ✓ gplearn_GP beats LinearRegression by {improvement:.2f}%")
        else:
            print(f"\n  ✗ LinearRegression beats gplearn_GP by {-improvement:.2f}%")

    print(f"\n{'='*60}")
    print(f"Results saved to: {result_dir}")
    print(
        "Files: mae_results.csv, gplearn_model.json, *_predictions.csv, predictions_comparison.png"
    )
    print(f"{'='*60}")