import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the CSV data into a pandas DataFrame
mental_health_data = pd.read_csv('D:\ML new\Deepression (1).csv')

# Display the first few and last few rows of the dataset
print(mental_health_data.head())
print(mental_health_data.tail())

# Display the number of rows and columns in the dataset
print(f"Shape of the dataset: {mental_health_data.shape}")

# Get some basic information about the dataset
mental_health_data.info()

# Check for missing values
print(f"Missing values in each column:\n{mental_health_data.isnull().sum()}")

# Display statistical measures of the dataset
print(mental_health_data.describe())

# Check the distribution of the 'Restlessness' column
print(f"Distribution of 'Restlessness' column:\n{mental_health_data['Restlessness'].value_counts()}")

# Check if the target column exists and find its distribution
target_column = 'Restlessness'

if target_column in mental_health_data.columns:
    target_distribution = mental_health_data[target_column].value_counts()
    print(f"Distribution of target column '{target_column}':\n{target_distribution}")
else:
    print(f"Column '{target_column}' does not exist in the dataset.")

# Check if the target column 'Restlessness' contains any missing values and drop rows with missing values
mental_health_data = mental_health_data.dropna(subset=['Restlessness'])

# Define feature matrix and target vector
X = mental_health_data.drop(columns='Restlessness', axis=1)
y = mental_health_data['Restlessness']

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"Categorical columns: {categorical_cols}")

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=categorical_cols)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Print the shape of the training and test sets
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# Initialize the LogisticRegression model
model = LogisticRegression(max_iter=1000)

# Train the LogisticRegression model with training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")

# Check accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)
print('Accuracy on Test data: ', test_data_accuracy)

# Example input data
input_data = (14, 2, 5, 5, 1, 1, 5, 1, 1, 5, 1, 1, 1, 1, 1)

# Convert input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Ensure the input data matches the shape of the training data
if input_data_as_numpy_array.shape[0] != X_train.shape[1]:
    raise ValueError(f"Expected input data of shape ({X_train.shape[1]},), but got {input_data_as_numpy_array.shape}")

# Reshape the input data to match the model's expected input shape
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Make a prediction
prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person does not have any disease')
else:
    print('The person has a disease')
