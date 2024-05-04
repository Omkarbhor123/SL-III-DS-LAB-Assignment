import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Load the dataset
data = pd.read_csv('/content/Iris.csv')

# Split the dataset into features and target
X = data.iloc[:, 1:-1].values  # Features
y = data.iloc[:, -1].values  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = gnb.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Compute error rate
error_rate = 1 - accuracy
print(f"Error Rate: {error_rate:.4f}")

# Compute precision
precision = precision_score(y_test, y_pred, average='macro')
print(f"Precision: {precision:.4f}")

# Compute recall
recall = recall_score(y_test, y_pred, average='macro')
print(f"Recall: {recall:.4f}")
