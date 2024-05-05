import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("/content/BostonHousing.csv")

# Impute missing values with the mean of each feature
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Convert the imputed data back to a DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Check for missing values after imputation
missing_values_after_imputation = data_imputed.isnull().sum()
print("Missing values after imputation:\n", missing_values_after_imputation)

# Split the data into features (X) and target variable (y)
X = data_imputed.drop("medv", axis=1)  # Features
y = data_imputed["medv"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Train the model using the scaled training sets
model.fit(X_train_scaled, y_train)

# Make predictions using the scaled testing set
y_pred = model.predict(X_test_scaled)

# Convert predicted and actual outputs to pandas DataFrame for easy formatting
results_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})

# Print predicted and actual output side by side
print(results_df)

# Calculate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", round(mse, 2))
print("R^2 Score:", round(r2, 2))
