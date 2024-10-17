# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
file_path = 'C:\\Users\\salee\\Downloads\\Task1\\german_credit_data.csv'
df = pd.read_csv(file_path)

# Check for missing values and column names
print("Columns in the dataset:", df.columns)
print("Missing values in each column:\n", df.isnull().sum())

# Create a copy of the dataframe
df_rf = df.copy()

# Handle missing values (fill with mode for categorical data)
df_rf['Saving accounts'] = df_rf['Saving accounts'].fillna(df_rf['Saving accounts'].mode()[0])
df_rf['Checking account'] = df_rf['Checking account'].fillna(df_rf['Checking account'].mode()[0])

# Label encoding for categorical columns
label_encoders = {}
for column in df_rf.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df_rf[column] = label_encoders[column].fit_transform(df_rf[column])

# Define features and target
target_column = 'Risk'
X = df_rf.drop(target_column, axis=1)
y = df_rf[target_column]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest classifier with class_weight='balanced' to handle class imbalance
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
rf_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)
