import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
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
##########################################################################
#Step 2: Data analysis and visualization
########################################
#2.1 Distribution of charges
y = df['charges']
plt.figure(figsize=(12, 6))

# First histogram
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
sns.set_style('whitegrid')
sns.histplot(y, bins=30, kde=True, color='pink', edgecolor='pink')
plt.xlabel('Charges')  # Set x label
plt.ylabel('Frequency')  # Set y label
plt.title('Distribution of Charges')

# Second histogram (log-transformed)
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
y_log = np.log(y)
sns.histplot(y_log, bins=30, kde=True, color='pink', edgecolor='pink')  # histogram of the log-transformed target
plt.xlabel('Log of Charges')  # Set x label
plt.ylabel('Frequency')  # Set y label
plt.title('Log-Transformed Distribution of Charges')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
#This shows that the distribution of charges is right-skewed, with a long tail of higher charges. The log transformation helps to normalize the distribution, making it more symmetric and easier to model with linear regression techniques.#We will use the log-transformed charges as our target vairable for modeling, as it is more normally distributed and can help improve the performance of regression models.

#2.2 Correlation bargraph and heatmap
corr = df.select_dtypes(include=[int,float]).corr() #get numeric columns

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
#print(corr['charges'])
#BarChart
plt.subplot(1, 2, 1) 
corr['charges'].drop('charges').plot(kind="barh",color='pink', edgecolor='pink')
plt.title('Correlation with Charges')

#Heatmap
plt.subplot(1, 2, 2) 
sns.heatmap(corr, annot=True, cmap='RdPu', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show() #The bar chart shows that 'bmi' and 'age' have the highest positive correlation with charges, while 'children' has a smaller positive correlation. The heatmap confirms these relationships and also shows that there are no strong negative correlations among the numeric features.
##########################################################################
#Step 3: Data preprocessing and feature engineering
###################################################
#3.1 Separate features and target variable for linear regression modeling
X_1 = df.select_dtypes(include=[int,float]).drop('charges', axis=1).copy() #all features
X_2 = X_1.drop('children',axis=1).copy() #drop 'children' since it has low correlation with charges
#y from y_log since we will use log-transformed charges as target variable
#3.2 Train-test split
X = X_1.drop('children',axis=1)
from sklearn.model_selection import train_test_split
X_train_1,X_val_1,y_train_log_1,y_val_log_1=train_test_split(X_1,y_log,test_size=0.2,random_state=42)
#3.3 Standardize numeric features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_1_sc = scaler.fit_transform(X_train_1)
X_val_1_sc = scaler.transform(X_val_1)


#scaling the features and target variable can help improve the performance of regression models by ensuring that all features are on a similar scale, which can help the model converge faster and perform better.


#########################################################
#4 Implement and evaluate linear regression model
#################################################

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
cv_score = cross_val_score(lr, X_train_1_sc, y_train_log_1, cv=5, scoring='neg_mean_squared_error')
RMSE =np.sqrt(-cv_score.mean())
print(f"Cross-validated RMSE: {RMSE:.2f}")


lr_model=lr.fit(X_train_1_sc, y_train_log_1)
y_pred_lr = lr_model.predict(X_val_1_sc)

plt.figure(figsize=(8, 6))
#plt.scatter(y_val_log_1, y_pred_lr, color='pink', edgecolor='pink')
#plt.scatter(np.expm1(y_val_log_1), np.expm1(y_pred_lr),
#color="pink", alpha=0.5, s=15)
#plt.plot([0, 800], [0, 800], 'r--',color="#B43757")

plt.scatter(y_val_log_1, y_pred_lr,  color='gray')
plt.plot([6,12],[6,12], color='red', linewidth=2)
plt.show()

"""The scatter plot of predicted vs actual log-transformed charges shows that the linear 
regression model is able to capture the general trend of the data, but there is some scatter 
around the ideal 45-degree line, indicating that the model is not perfectly accurate. 
The RMSE value of 0.77 indicates that on average, the model's predictions are off by about 0.77 in 
log-transformed charge units, which suggests that there is room for improvement in the model's performance."""
#What if we remove the 'children' feature since it has low correlation with charges?
X_train_2,X_val_2,y_train_log_2,y_val_log_2=train_test_split(X_2,y_log,test_size=0.2,random_state=42)

X_train_2_sc = scaler.fit_transform(X_train_2)
X_val_2_sc = scaler.transform(X_val_2)

lr_model_2=lr.fit(X_train_2_sc, y_train_log_2)
y_pred_lr_2 = lr_model_2.predict(X_val_2_sc)

score_2 = cross_val_score(lr, X_train_2_sc, y_train_log_2, cv=5, scoring='neg_mean_squared_error')
RMSE_2 =np.sqrt(-score_2.mean())
print(f"Cross-validated RMSE without 'children': {RMSE_2:.2f}")
plt.figure(figsize=(8, 6))
plt.scatter(y_val_log_2, y_pred_lr_2,  color='gray')
plt.plot([6,12],[6,12], color='red', linewidth=2)
plt.show()

"""Removing the 'children' feature did not significantly improve the model's performance, as the RMSE 
only increased slightly from 0.77 to 0.78. The scatter plot of predicted vs actual log-transformed charges 
still shows a similar pattern, indicating that the 'children' feature does not have a strong impact on 
the model's ability to predict charges. Therefore, it may be reasonable to keep the 'children' feature in 
the model for interpretability purposes, even though it does not contribute much to predictive performance."""


#########################################################
#5 Implement and evaluate logistic regression model
###################################################