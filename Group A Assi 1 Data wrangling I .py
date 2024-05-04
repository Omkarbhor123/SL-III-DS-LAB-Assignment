
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1. Load the Dataset into pandas DataFrame
iris_data = pd.read_csv("/content/Iris.csv")

# 2. Data Preprocessing

# Check for missing values
missing_values = iris_data.isnull().sum()
print("Missing values:\n", missing_values)

# Dataset description
print("\nDataset description:\n", iris_data.describe())

# Variable descriptions
print("\nVariable descriptions:")
print("Id: Identifier for each observation")
print("SepalLengthCm: Length of sepal in centimeters")
print("SepalWidthCm: Width of sepal in centimeters")
print("PetalLengthCm: Length of petal in centimeters")
print("PetalWidthCm: Width of petal in centimeters")
print("Species: Species of iris flower (categorical variable)")

# Check the dimensions of the DataFrame
print("\nDataFrame dimensions:", iris_data.shape)

# 3. Data Formatting and Data Normalization

# Check the data types of variables
print("\nData types:\n", iris_data.dtypes)

# 4. Turn categorical variables into quantitative variables

# Create a label encoder object
label_encoder = LabelEncoder()

# Encode the 'Species' column
iris_data['Species'] = label_encoder.fit_transform(iris_data['Species'])

# Display the encoded 'Species' column
print("\nSpecies column after label encoding:\n", iris_data['Species'].value_counts())
