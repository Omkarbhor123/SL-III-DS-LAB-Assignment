import pandas as pd
import numpy as np

# Load the dataset
academic_performance = pd.read_csv('/content/Academic performance.csv')

# 1. Scan all variables for missing values and inconsistencies, and deal with them
print("Missing values:\n", academic_performance.isnull().sum())

# Fill missing values with the mean/mode/median of the respective columns
academic_performance['Age'] = academic_performance['Age'].fillna(academic_performance['Age'].mean())
academic_performance['Gender'] = academic_performance['Gender'].fillna(academic_performance['Gender'].mode()[0])
academic_performance['Grade'] = academic_performance['Grade'].fillna(academic_performance['Grade'].median())
academic_performance['Attendance'] = academic_performance['Attendance'].fillna(academic_performance['Attendance'].median())

# Check for inconsistent values in 'Gender' column
print("\nInconsistent values in 'Gender' column:\n", academic_performance['Gender'].value_counts())

# Replace inconsistent values with 'M' or 'F'
academic_performance['Gender'] = academic_performance['Gender'].replace(['m', 'f'], ['M', 'F'])

# 2. Scan numeric variables for outliers and deal with them
print("\nOutliers in numeric variables:")
for col in academic_performance.select_dtypes(include=['number']).columns:
    print(f"\n{col}:")
    print(academic_performance[col].describe())

# Replace the outlier value in 'Stress Level' with the median
academic_performance['Stress Level'] = academic_performance['Stress Level'].replace(999, academic_performance['Stress Level'].median())

# 3. Apply data transformations on at least one variable
# Apply log transformation to 'Grade' to decrease skewness
academic_performance['Grade_log'] = np.log(academic_performance['Grade'])

# Check skewness before and after transformation
print("\nSkewness of 'Grade' before transformation:", academic_performance['Grade'].skew())
print("Skewness of 'Grade_log' after transformation:", academic_performance['Grade_log'].skew())
