import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
dataset = pd.read_csv('/content/Academic performance.csv')

# 1. Scan all variables for missing values and inconsistencies, and deal with them
print("Missing values:\n", dataset.isnull().sum())

# Fill missing values with the mean/mode/median of the respective columns
dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
dataset['Grade'] = dataset['Grade'].fillna(dataset['Grade'].median())
dataset['Attendance'] = dataset['Attendance'].fillna(dataset['Attendance'].median())

# Check for inconsistent values in 'Gender' column
print("\nInconsistent values in 'Gender' column:\n", dataset['Gender'].value_counts())

# Replace inconsistent values with 'M' or 'F'
dataset['Gender'] = dataset['Gender'].replace(['m', 'f'], ['M', 'F'])

# 2. Scan numeric variables for outliers and deal with them
print("\nOutliers in numeric variables:")

numeric_cols = dataset.select_dtypes(include=['number']).columns

# Create a box plot for each numeric column
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dataset, x=col)
    plt.title(f"Box Plot for {col}")
    plt.show()

# Identify and remove outliers based on box plot visualization
for col in numeric_cols:
    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)
    dataset = dataset[(dataset[col] >= lower_bound) & (dataset[col] <= upper_bound)]

# Print the remaining data after removing outliers
print("\nData after removing outliers:\n", dataset)

# 3. Apply data transformations on at least one variable
# Apply log transformation to 'Grade' to decrease skewness
dataset['Grade_log'] = np.log(dataset['Grade'])

# Check skewness before and after transformation
print("\nSkewness of 'Grade' before transformation:", dataset['Grade'].skew())
print("Skewness of 'Grade_log' after transformation:", dataset['Grade_log'].skew())
