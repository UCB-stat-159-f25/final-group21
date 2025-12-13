import pandas as pd
import numpy as np
import kagglehub
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, roc_curve, auc

import dataframe_image as dfi

def get_full_data():
    """
    This function extracts the raw data from Kaggle:
    https://www.kaggle.com/datasets/usdot/flight-delays/

    Parameters
    ----------
    None
    
    Returns
    -------
    str
        File path to the downloaded dataset directory.
    """
    
    path = kagglehub.dataset_download("usdot/flight-delays")
    
    return path


def summarize_df(df, output_path=None):
    """
    This function creates a summary of a given pandas dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe to be summarized.
    output_path : str
        String containing the output path of where the figure should be saved.

    Returns
    -------
    pandas.DataFrame
        A summary table containing metrics for each column.
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

    # Specifying column order for dfi to work
    col_order = ["dtype", "min", "max", "mean", "std", "unique_vals", "missing_pct"]
    summary = summary[col_order]

    if output_path:
        dfi.export(summary, output_path, table_conversion="matplotlib")

    return summary


def group_data_reliability(data, groupby):
    """
    This function groups data for reliability by a certain subject.

    Parameters
    ----------
    data : pandas.DataFrame
        Flight-level dataset containing delay and on-time indicators.
    groupby : str
        Column name to group by (e.g., airline or origin airport).

    Returns
    -------
    pandas.DataFrame
        Summary table with average departure and arrival delays, on-time
        rates, and total number of flights for each group.
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

    Parameters
    ----------
    summary_performance : pandas.DataFrame
        Summary table containing abbreviated subject identifiers.
    ext_data : pandas.DataFrame
        External dataset containing full subject names and IATA codes.
    left_merge : str
        Column name in summary_performance used for merging.
    drop_columns : list of str
        List of column names to drop after the merge.

    Returns
    -------
    pandas.DataFrame
        Summary table with full subject names merged in and specified
        columns removed.
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

    Parameters
    ----------
    summary_performance : pandas.DataFrame
        Summary table containing on-time rates and average delay metrics.

    Returns
    -------
    pandas.DataFrame
        Sorted summary table ordered by arrival and departure
        on-time rate, and average arrival and departure delay.
    """
        
    summary_performance_out = summary_performance.sort_values(
        by=["arr_on_time_rate", "dep_on_time_rate", "mean_arr_delay", "mean_dep_delay"],
        ascending=[False, False, True, True]
    )

    return summary_performance_out


def on_time_summary_table(summary_performance, subjects, subjects_column_pretty, output_path=None):
    """
    This function creates a summary table containing containing the subjects and each of their metrics.

    Parameters
    ----------
    summary_performance : pandas.DataFrame
        Summary table containing reliability metrics for each subject.
    subjects : str
        Column name containing the subjects (e.g., airline or airport).
    subjects_column_pretty : str
        Pretty-formatted column name used in the output table.
    output_path : str
        String containing the output path of where the figure should be saved.

    Returns
    -------
    pandas.DataFrame
        Formatted summary table with subject names, total number of flights,
        on-time rates, and average delay metrics.
    """
    
    table = summary_performance[[subjects, "n_flights",
                           "arr_on_time_rate", "dep_on_time_rate",
                           "mean_arr_delay", "mean_dep_delay"]
                           ].rename(columns={subjects: subjects_column_pretty,
                                             "n_flights": "Total Flights",
                                             "arr_on_time_rate": "On Time Arrivals (Rate)",
                                             "dep_on_time_rate": "On Time Departures (Rate)",
                                             "mean_arr_delay": "Average Arrival Delay (Minutes)",
                                             "mean_dep_delay": "Average Departure Delay (Minutes)"}
                                   ).reset_index(drop=True)

    # Specifying column order for dfi to work
    col_order = [subjects_column_pretty, "Total Flights", "On Time Arrivals (Rate)", "On Time Departures (Rate)", 
                  "Average Arrival Delay (Minutes)", "Average Departure Delay (Minutes)"]
    table = table[col_order]

    if output_path:
        dfi.export(table, output_path, table_conversion="matplotlib")

    return table


def on_time_bar_chart(subjects, summary_performance, output_path=None):
    """
    This function creates a bar chart showing both the on time rate and the average delays
    for each of the subjects.

    Parameters
    ----------
    subjects : str
        Column name in summary_performance containing the subject labels
        (e.g., airline codes or airport).
    summary_performance : pandas.DataFrame
        Summary table containing on-time rates and average delay metrics
        for each subject.
    output_path : str
        String containing the output path of where the figure should be saved.

    Returns
    -------
    None
        Displays the bar chart and saves it, but does not return any value.
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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    return None


def plot_corr_matrix(corr_matrix, title, output_path=None):
    """
    This function plots an already existing correlation matrix.

    Parameters
    ----------
    corr_matrix : pandas.DataFrame
        Square correlation matrix containing correlation coefficients.
    title : str
        Title to be displayed above the correlation plot.
    output_path : str
        String containing the output path of where the figure should be saved.

    Returns
    -------
    None
        Displays the correlation heatmap and saves it, but does not return any value.
    """
    
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix)
    color = plt.imshow(corr_matrix, vmin=-1, vmax=1, cmap="bwr")
    plt.colorbar(color)
    
    plt.xticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha="right")
    plt.yticks(np.arange(len(corr_matrix.columns)), corr_matrix.columns)
    
    plt.title(title)
    
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    return None


def group_data_clustering(data, groupby, keep_cols):
    """
    This function groups data for clustering by a certain subject.

    Parameters
    ----------
    data : pandas.DataFrame
        Flight-level dataset containing all variables used for clustering.
    groupby : str
        Column name to group by (e.g., airline or origin airport).
    keep_cols : list of str
        List of column names to retain for clustering aggregation.

    Returns
    -------
    pandas.DataFrame
        Summary table containing metrics for each group.
    """
    
    summary = (
        data.groupby(groupby)[keep_cols]
            .agg(["mean", 
                  "std", 
                  lambda x: x.quantile(0.1),
                  lambda x: x.quantile(0.9)])
    )

    return summary


def hierarchical_clustering(data, labels, title, output_path=None):
    """
    This function performs hierarchical clustering and plots the corresponding dendrogram.

    Parameters
    ----------
    data : numpy.ndarray
        Numeric matrix to be clustered.
    labels : list of str
        Labels corresponding to each row in data used for dendrogram leaf names.
    title : str
        Title to be displayed above the dendrogram.
    output_path : str
        String containing the output path of where the figure should be saved.

    Returns
    -------
    None
        Displays the dendrogram and does not return any value.
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

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    return None


def preprocessor(X_train, categorical_variables):
    """
    This function defines the preprocessing necessary for training the models.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training dataset containing both numeric and categorical variables.
    categorical_variables : list of str
        List of column names in X_train that should be treated as categorical
        and one-hot encoded.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A preprocessing object that applies passthrough to numeric variables
        and one-hot encoding to categorical variables.
    """
    
    # Defining the categorical and numeric variables in the data
    categorical_variables = categorical_variables
    numeric_variables = [col for col in X_train.columns if col not in categorical_variables]
    
    # Defining the transformers of the categorical (one hot encoding) and numeric variables (none)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_variables),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_variables)
        ]
    )
    
    return preprocessor


def train_models(model_type, models, preprocessor, X_train, y_train, X_test, y_test):
    """
    This functions trains the regression or classification models specified.

    Parameters
    ----------
    model_type : str
        Type of models to train. Must be either "regression" or "classification".
    models : dict
        Dictionary of model names mapped to scikit-learn model objects.
    preprocessor : sklearn.compose.ColumnTransformer
        Preprocessing pipeline applied to the data before model training.
    X_train : pandas.DataFrame
        Training dataset.
    y_train : pandas.Series
        Training target variable.
    X_test : pandas.DataFrame
        Test dataset.
    y_test : pandas.Series
        Test target variable.

    Returns
    -------
    dict
        Dictionary containing model evaluation results. For regression models,
        the output contains RMSE values for each model. For classification models,
        the output contains ROC-AUC values and corresponding false positive rates,
        true positive rates, and thresholds for each model.
    """
    
    # Defining masks for the data used for regression models, filtering out undefined values
    if model_type == "regression":
        mask_train = (y_train.notna())
        mask_test = (y_test.notna())

        X_train = X_train[mask_train]
        y_train = y_train[mask_train]
        X_test = X_test[mask_test]
        y_test = y_test[mask_test]
    
    results = {}
    
    for name, model in tqdm(models.items(), desc="Training models"):
        # Defining the pipeline used to preprocess and train the model
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
    
        # Fitting the model on training data and computing predictions for the test data
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        pipeline.fit(X_train, y_train)
        
        if model_type == "regression":
            y_pred = pipeline.predict(X_test)
            
            # Computing the RMSE between predictions and actual values on the test data
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            results[name] = rmse
        
        else:
            y_pred = pipeline.predict_proba(X_test)[:, 1]
            
            # Computing ROC AUC metrics for plotting and evaluating
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
    
            # Saving the results
            results[name] = {
                "roc_auc": roc_auc,
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds
            }

    return results


def roc_auc_plot(results_cla, output_path=None):
    """
    This function plots a ROC-AUC curve based on the given results.

    Parameters
    ----------
    results_cla : dict
        Dictionary containing classification results for each model.
    output_path : str
        String containing the output path of where the figure should be saved.

    Returns
    -------
    None
        Displays the ROC-AUC plot and does not return any value.
    """
    
    # Plotting the ROC-AUC
    plt.figure()
    
    for name, metrics in results_cla.items():
        plt.plot(
            metrics["fpr"],
            metrics["tpr"],
            label=f"{name} (AUC = {metrics['roc_auc']:.3f})"
        )
    
    # Plotting baseline
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random", color="black")
    
    # Names and legend
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curves")
    plt.legend()
    
    # Show plot
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    return None


#Convert HHMM time into hour of day
"""
Parameters x: int, float, or str
Returns: float, hour of day (0-23), or NaN if missing.
"""
def hhmm_to_hour(x):
    if pd.isna(x):
        return np.nan
    try:
        x = int(x)
    except Exception:
        return np.nan

    hour = x // 100
    minute = x % 100

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return np.nan

    return hour


