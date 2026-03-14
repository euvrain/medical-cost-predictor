import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import streamlit
import joblib

#Step 1: Load and explore data

df = pd.read_csv("/Users/taylormcdonald/medical-cost-predictor/data/medical-charges.csv")

print(f"Info: {df.info()}\n") #print data column names, data types, and non-null counts
print("Data type counts:")
for data_type in df.dtypes.value_counts().index:
    print(f"{data_type}: {df.dtypes.value_counts()[data_type]}") #for each data type, clearly print the count of columns with that data type

print(f"\nDescription: \n{df.describe()}") #print summary statistics for numeric columns (count, mean, std, min, 25%, 50%, 75%, max)
#print(f"\n# of null: {df.isnull().sum()}")

#Step 2: Data analysis and visualization
#Correlation histogram and heatmap

plt.hist(df[:-1].corr(df['charges']), bins=30, edgecolor='black')