import pandas as pd
import numpy as np
import kagglehub

def get_full_data():
	"""
	This function extracts the raw data from Kaggle:
	https://www.kaggle.com/datasets/usdot/flight-delays/
	"""
	
	path = kagglehub.dataset_download("usdot/flight-delays")
	
	return path

def summarize_df(df):
	"""
	This function creates a summary of a given pandas dataframe
	"""
	
	summary = pd.DataFrame({
	    "dtype": df.dtypes,
	    "min": df.min(numeric_only=True),
	    "max": df.max(numeric_only=True),
	    "mean": df.mean(numeric_only=True),
	    "std": df.std(numeric_only=True),
	    "unique_vals": df.nunique(),
	    "missing_pct": df.isna().mean()
    })

	return summary


