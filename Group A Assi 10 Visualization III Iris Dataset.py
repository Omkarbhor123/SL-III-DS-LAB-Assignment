import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load Iris dataset
iris = load_iris()

# Convert to DataFrame
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add target column
iris_df['species'] = iris.target

# Map target values to species names
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 1. Features and their types
features_and_types = iris_df.dtypes
print("Features and their types:")
print(features_and_types)

# 2. Histogram for each feature
iris_df.hist(figsize=(12, 8))
plt.suptitle('Histograms of Iris Dataset Features')
plt.show()

# 3. Boxplot for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=iris_df)
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# 4. Outlier identification
plt.figure(figsize=(12, 8))
for i, feature in enumerate(iris.feature_names):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=iris_df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()


# Observations:

# Features and Their Types:
# Sepal length (numeric)
# Sepal width (numeric)
# Petal length (numeric)
# Petal width (numeric)
# Histograms:
# Sepal length and width seem to have somewhat normal distributions.
# Petal length and width show clear distinctions between different groups or clusters.
# Boxplots:
# Boxplots provide a visual summary of each feature's distribution across different species.
# They show the spread and central tendency of each feature for each species.
# Sepal width appears to have the most variation among the three species.
# Petal length and width have clear distinctions between different species, indicating their potential for species classification.
# Outliers:
# Outliers can be identified in the boxplots as individual points that fall far away from the main distribution.
# For example, there are a few outliers in sepal width and petal width. These outliers might be worth investigating further as they could represent measurement errors or unique specimens.
