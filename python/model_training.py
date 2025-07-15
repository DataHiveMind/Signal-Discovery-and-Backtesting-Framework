import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, roc_auc_score


def train_ridge(
    df: pd.DataFrame,
    target_col: str = 'future_return',
    alpha: float = 1.0
):
    """
    Fit a Ridge regression model on standardized features.

    Args:
        df: DataFrame of features including target_col.
        target_col: Column name for regression target.
        alpha: Regularization strength.

    Returns:
        Trained Ridge model, X_test, y_test.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train/test split: last 20% as test
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)

    return {
        'model': model,
        'scaler': scaler,
        'X_test': X_test_scaled,
        'y_test': y_test.values
    }


def evaluate_model(
    model_dict: dict
) -> dict:
    """
    Compute evaluation metrics for regression and classification.

    Args:
        model_dict: Output of train_ridge or similar with model, X_test, y_test.

    Returns:
        Dict containing MSE, RMSE, and optionally AUC for binary.
    """
    model = model_dict['model']
    X_test = model_dict['X_test']
    y_test = model_dict['y_test']
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    results = {'mse': mse, 'rmse': rmse}

    # If binary target, compute AUC
    if set(np.unique(y_test)) <= {0, 1}:
        results['auc'] = roc_auc_score(y_test, preds)

    return results
