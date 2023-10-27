import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset (replace 'your_dataset.csv' with the actual dataset)
df = pd.read_csv('your_dataset.csv')

# Specify dependent and independent variables
X = df[['Road_Conditions', 'Visibility', 'Speed_Limit', 'Num_Vehicles', 'Time_of_Day']]
y = df['Accident_Severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model for future use
joblib.dump(model, 'accident_severity_model.joblib')

# Evaluate the model on the test set (optional)
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy: {accuracy * 100:.2f}%')