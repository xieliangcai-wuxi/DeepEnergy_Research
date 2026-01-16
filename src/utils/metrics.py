import numpy as np

def calculate_metrics(preds: np.ndarray, trues: np.ndarray):
    """
    计算时间序列预测的常用指标。
    Args:
        preds: 预测值 (N, )
        trues: 真实值 (N, )
    """
    # 防止除以0
    mask = trues != 0
    
    mae = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    
    # MAPE 往往受极小值影响，这里加一个 epsilon 或者只计算非零处
    mape = np.mean(np.abs((preds[mask] - trues[mask]) / trues[mask])) * 100
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape
    }