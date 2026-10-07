import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('full_graded_dataset.csv')

# Replace 'human_score' with your actual column name if present
if 'human_score' in df.columns:
    y_true = df['human_score']
    y_pred = df['final_grade_out_of_10']

    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"📊 Model Evaluation Metrics:")
    print(f"• Pearson Correlation (r) : {pearson_corr:.4f}")
    print(f"• Spearman Correlation   : {spearman_corr:.4f}")
    print(f"• Mean Absolute Error    : {mae:.4f}")
    print(f"• Root Mean Squared Error: {rmse:.4f}")
else:
    print("⚠️ Add a 'human_score' column to compute validation metrics.")