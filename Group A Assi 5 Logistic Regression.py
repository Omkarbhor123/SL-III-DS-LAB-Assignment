import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
data = pd.read_csv('Social_Network_Ads.csv')

# Let's assume 'Age', 'EstimatedSalary' as features and 'Purchased' as target
X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize logistic regression model
classifier = LogisticRegression(random_state=42)

# Fit the model
classifier.fit(X_train, y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract TP, FP, TN, FN
TP = cm[1][1]
FP = cm[0][1]
TN = cm[0][0]
FN = cm[1][0]

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)

# Compute error rate
error_rate = 1 - accuracy

# Compute precision
precision = precision_score(y_test, y_pred)

# Compute recall
recall = recall_score(y_test, y_pred)

print()
print("Performance Metrics :")
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)


# Predicted
#          0    1
# Actual 0 TN   FP
#        1 FN   TP

# Here's how to interpret a confusion matrix:

# True Positive (TP): The number of instances that were correctly predicted as positive (Purchased in this case).
# False Positive (FP): The number of instances that were incorrectly predicted as positive. These are instances that were actually negative (not Purchased), but the model predicted them as positive.
# True Negative (TN): The number of instances that were correctly predicted as negative (not Purchased).
# False Negative (FN): The number of instances that were incorrectly predicted as negative. These are instances that were actually positive (Purchased), but the model predicted them as negative.
