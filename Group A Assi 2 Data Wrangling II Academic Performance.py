import pandas as pd
import numpy as np

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
for col in dataset.select_dtypes(include=['number']).columns:
    print(f"\n{col}:")
    print(dataset[col].describe())

# Replace the outlier value in 'Stress Level' with the median
dataset['Stress Level'] = dataset['Stress Level'].replace(999, dataset['Stress Level'].median())

# 3. Apply data transformations on at least one variable
# Apply log transformation to 'Grade' to decrease skewness
dataset['Grade_log'] = np.log(dataset['Grade'])

# Check skewness before and after transformation
print("\nSkewness of 'Grade' before transformation:", dataset['Grade'].skew())
print("Skewness of 'Grade_log' after transformation:", dataset['Grade_log'].skew())

# import matplotlib.pyplot as plt
# plt.boxplot(dataset['Stress Level'])
# plt.show()
