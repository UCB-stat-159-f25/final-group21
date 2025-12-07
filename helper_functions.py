import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

def get_full_data():
	"""
	This function extracts the raw data from Kaggle:
	https://www.kaggle.com/datasets/usdot/flight-delays/
	"""
	
	path = kagglehub.dataset_download("usdot/flight-delays")
	
	return path


def summarize_df(df):
	"""
	This function creates a summary of a given pandas dataframe.
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


def group_data_reliability(data, groupby):
	"""
	This function groups data for reliability by a certain subject.
	"""
	
	summary_performance = (
		data.groupby(groupby)
			.agg(
				mean_dep_delay=("DEPARTURE_DELAY", "mean"),
				mean_arr_delay=("ARRIVAL_DELAY", "mean"),
				dep_on_time_rate=("dep_on_time", "mean"),
				arr_on_time_rate=("arr_on_time", "mean"),
				n_flights=(groupby, "size")
			)
	)
	
	return summary_performance

def show_full_names(summary_performance, ext_data, left_merge, drop_columns):
	"""
	This function merges data with an external datasource to match the full subject names
	to the abbreviations in the original data.
	"""
	
	summary_performance_out = summary_performance.merge(
		ext_data,
		left_on=left_merge,
		right_on="IATA_CODE",
		how="left"
	).drop(columns=drop_columns)

	return summary_performance_out


def sort_on_time_rate(summary_performance):
	"""
	This function sorts a summary dataset on on time rate and thereafter on average delay.
	"""
    
	summary_performance_out = summary_performance.sort_values(
		by=["arr_on_time_rate", "dep_on_time_rate", "mean_arr_delay", "mean_dep_delay"],
		ascending=[False, False, True, True]
	)

	return summary_performance_out


def on_time_summary_table(summary_performance, subjects, subjects_column, subjects_column_pretty):
	"""
	This function creates a summary table containing containing the subjects and each of their metrics.
	"""
	
	table = summary_performance[[subjects, "n_flights",
						   "arr_on_time_rate", "dep_on_time_rate",
						   "mean_arr_delay", "mean_dep_delay"]
						   ].rename(columns={subjects_column: subjects_column_pretty,
											 "n_flights": "Total Flights",
											 "arr_on_time_rate": "On Time Arrivals (Rate)",
											 "dep_on_time_rate": "On Time Departures (Rate)",
											 "mean_arr_delay": "Average Arrival Delay (Minutes)",
											 "mean_dep_delay": "Average Departure Delay (Minutes)"}
								   ).reset_index(drop=True)

	return table


def on_time_bar_chart(subjects, summary_performance):
	"""
	This function creates a bar chart showing both the on time rate and the average delays
	for each of the subjects.
	"""
	
	# Showing the ranked subjects in a bar chart
	fig, ax1 = plt.subplots(figsize=(14, 8))
	
	# On Time Rates
	width = 0.35
	ax1.bar(np.arange(len(summary_performance[subjects])) - width/2, 
			summary_performance["arr_on_time_rate"], width, label="On Time Arrivals (Rate)")
	ax1.bar(np.arange(len(summary_performance[subjects])) + width/2, 
			summary_performance["dep_on_time_rate"], width, label="On Time Departures (Rate)")
	ax1.set_ylabel("On Time Rate")
	ax1.set_ylim(0, 1)
	ax1.set_xticks(np.arange(len(summary_performance[subjects])))
	ax1.set_xticklabels(summary_performance[subjects], rotation=45, ha="right")
	
	# Average Delays
	ax2 = ax1.twinx()
	ax2.plot(np.arange(len(summary_performance[subjects])), 
			 summary_performance["mean_dep_delay"], marker="s", markersize=15, 
			 linestyle="None", label="Average Departure Delay (Minutes)", 
			 markerfacecolor="white", markeredgecolor="black")
	ax2.plot(np.arange(len(summary_performance[subjects])), 
			 summary_performance["mean_arr_delay"], marker="o", markersize=15, 
			 linestyle="None", label="Average Arrival Delay (Minutes)", 
			 markerfacecolor="white", markeredgecolor="black")
	ax2.set_ylim(-5, 40)
	ax2.set_ylabel("Average Delay")
	
	# Title
	fig.suptitle("On Time Rates and Average Delays")
	
	# Legend
	handles1, labels1 = ax1.get_legend_handles_labels()
	handles2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
	
	plt.tight_layout()
	plt.show()

	return None


def plot_corr_matrix(corr_matrix, title):
	"""
	This function plots an already existing correlation matrix.
	"""
	
	plt.figure(figsize=(10, 8))
	plt.imshow(corr_matrix)
	color = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="bwr")
	plt.colorbar(color)
	
	plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
	plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
	
	plt.title(title)
	
	plt.tight_layout()
	plt.show()

	return None


def group_data_clustering(data, groupby, keep_cols):
	"""
	This function groups data for clustering by a certain subject.
	"""
	
	summary = (
		data.groupby(groupby)[keep_cols]
			.agg(["mean", 
				  "std", 
				  lambda x: x.quantile(0.1),
				  lambda x: x.quantile(0.9)])
	)

	return summary


def hierarchical_clustering(data, labels, title):
	"""
	This function performs hierarchical clustering and plots the corresponding dendrogram.
	"""
	
	# Standard scaling the variables in the dataset
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(data)

	# Defining the linkage method used in the clustering
	Z = linkage(X_scaled, method="ward")

	# Plotting the dendrogram
	plt.figure(figsize=(12, 8))
	dendrogram(Z, labels=labels, leaf_rotation=45)
	plt.xticks(rotation=45, ha="right")
	plt.title(title)
	plt.ylabel("Distance")
	
	plt.tight_layout()
	plt.show()

	return None
