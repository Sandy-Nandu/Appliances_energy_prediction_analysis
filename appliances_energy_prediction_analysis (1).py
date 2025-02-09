
# # Appliances Energy Prediction - Data Analysis

# ### Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ### Loading the dataset
file_path = 'C:/Users/kavya/Downloads/appliances+energy+prediction/energydata_complete.csv'
data = pd.read_csv(file_path)

# ### Displaying the first few rows of the dataset
data.head()

# ### Checking for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# ### Data types and general information
data.info()

# ### Descriptive statistics
data.describe()

# ### Correlation Matrix
plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# ### Selecting Features for Regression
# Target: 'Appliances' (energy use)
# Features: Temperature and Humidity values along with other environment factors
X = data[['T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6']]
y = data['Appliances']

# ### Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### Building a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# ### Making Predictions on the Test Set
y_pred = model.predict(X_test)

# ### Evaluating the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# ### Plotting Predictions vs Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Energy Use (Appliances)')
plt.ylabel('Predicted Energy Use')
plt.title('Predicted vs Actual Energy Use')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.show()
