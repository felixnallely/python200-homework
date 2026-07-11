#---- Pipelines ----
#--- Pipeline Question 2 ---
import pandas as pd
import numpy as np 
from prefect import flow, task 

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

@task 
def create_series(input_arr: np.ndarray) -> pd.Series:
    return pd.Series(input_arr, name="values")

@task 
def clean_data(series: pd.Series) -> pd.Series:
    return series.dropna()

@task 
def summarize_data(series: pd.Series) -> dict:
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "mode": float(series.mode()[0]),
    }

#-- Perfect Flow --
@flow
def pipeline_flow():
    s = create_series(arr)
    cleaned = clean_data(s)
    summary = summarize_data(cleaned)
    return summary

if __name__ == "__main__":
    pipeline_flow()

#---- comment answer ----
#- Why might Prefect be more overhead than it is worth here?
#  Because this question deals with a small array and using Prefect adds additional work when initializing and in execution time unlike using plain Python calls.
#
#- Logical scenarios where framework like Prefect could be useful:
#  -When having team collabrations since it can give other members access to any faliures on a project, etc. 
#  -When scheduling since the workflow is small and can be run every hour, day and week. 
