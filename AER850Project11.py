#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:46:41 2024

@author: joe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:53:12 2024

@author: joe
"""
# Step 1: Data Processing
import pandas as pd

# Load the dataset from CSV
data = pd.read_csv("/Users/joe/Downloads/Project_1_Data (1).csv")

# Display the first few rows to understand the data
print(data.head())

# Step 2: Data Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (if not already loaded)
# data = pd.read_csv("Project_1_Data.csv")

# -------------------- Statistical Analysis --------------------
# Summary statistics for the dataset
print("Statistical Summary of the Dataset:")
print(data.describe())

# Display the distribution of the target variable (Maintenance Step)
plt.figure(figsize=(8,5))
sns.countplot(x='Step', data=data, palette='Set2')
plt.title('Distribution of Maintenance Steps')
plt.xlabel('Maintenance Step')
plt.ylabel('Count')
plt.show()

# -------------------- Data Visualization --------------------
# Visualize the relationship between coordinates (X, Y, Z) and the target variable (Step)

# Pairplot to visualize the relationship between features and target variable
plt.figure(figsize=(10,8))
sns.pairplot(data, hue='Step', markers=["o", "s", "D"], palette='coolwarm')
plt.suptitle('Pairplot of Coordinates (X, Y, Z) and Maintenance Steps', y=1.02)
plt.show()

# 3D Scatter plot to visualize X, Y, Z coordinates and how they relate to maintenance steps
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(data['X'], data['Y'], data['Z'], c=data['Step'], cmap='coolwarm', s=50)

# Adding axis labels
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('3D Scatter Plot of Coordinates (X, Y, Z) by Maintenance Step')

# Adding a legend
legend = ax.legend(*scatter.legend_elements(), title="Maintenance Step")
ax.add_artist(legend)

plt.show()

# -------------------- Explanation of Findings --------------------
# Statistical Summary Explanation:
print("\nExplanation of Findings:")
print("""
The statistical summary provides key insights into the distribution of the data:
- The 'describe()' function shows that the dataset is well-structured with no missing values.
- The count plot indicates the distribution of each maintenance step, which appears relatively balanced across most steps, though some steps have slightly fewer samples.
- In the pairplot, we can observe some degree of separation between classes (maintenance steps) based on the coordinates (X, Y, Z), suggesting that the data may be separable using machine learning models.
- The 3D scatter plot further highlights this separation, with distinct clusters for different maintenance steps. However, there is some overlap between certain steps, indicating that a more advanced classifier may be needed to achieve high accuracy.
""")


## Step 3: Correlation Analysis using Pearson Correlation
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the Pearson correlation matrix
pearson_corr_matrix = data[['X', 'Y', 'Z', 'Step']].corr(method='pearson')

# Plot a heatmap to visualize the Pearson correlation
plt.figure(figsize=(8,6))
sns.heatmap(pearson_corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Pearson Correlation Matrix")
plt.show()

#step 4
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np

# Split the dataset into features (X) and target (y)
X = data[['X', 'Y', 'Z']]  # Features: Coordinates
y = data['Step']  # Target: Maintenance step

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}")
    return y_pred

# -------------------- Classification Model 1: Random Forest with GridSearchCV --------------------
print("\nRandom Forest Classifier with GridSearchCV:")
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Best Random Forest model
best_rf = grid_rf.best_estimator_
print(f"Best Hyperparameters for Random Forest: {grid_rf.best_params_}")
evaluate_model(best_rf, X_test, y_test)

# -------------------- Classification Model 2: Support Vector Classifier with GridSearchCV --------------------
print("\nSupport Vector Classifier (SVC) with GridSearchCV:")
svc = SVC(random_state=42)
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_svc = GridSearchCV(svc, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1)
grid_svc.fit(X_train, y_train)

# Best SVC model
best_svc = grid_svc.best_estimator_
print(f"Best Hyperparameters for SVC: {grid_svc.best_params_}")
evaluate_model(best_svc, X_test, y_test)

# -------------------- Classification Model 3: K-Nearest Neighbors (KNN) with GridSearchCV --------------------
print("\nK-Nearest Neighbors (KNN) with GridSearchCV:")
knn = KNeighborsClassifier()
param_grid_knn = {
    'n_neighbors': np.arange(1, 15),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # p=1 for Manhattan distance, p=2 for Euclidean distance
}
grid_knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_knn.fit(X_train, y_train)

# Best KNN model
best_knn = grid_knn.best_estimator_
print(f"Best Hyperparameters for KNN: {grid_knn.best_params_}")
evaluate_model(best_knn, X_test, y_test)

# -------------------- Classification Model 4: RandomizedSearchCV with Random Forest --------------------
print("\nRandom Forest Classifier with RandomizedSearchCV:")
param_dist_rf = {
    'n_estimators': np.arange(10, 200, 10),
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': np.arange(2, 20, 2)
}
random_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_rf.fit(X_train, y_train)

# Best Random Forest model using RandomizedSearchCV
best_random_rf = random_rf.best_estimator_
print(f"Best Hyperparameters for Random Forest (RandomizedSearchCV): {random_rf.best_params_}")
evaluate_model(best_random_rf, X_test, y_test)

#step 5

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluate and compare the models using accuracy, F1 score, and precision
def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}")
    return y_pred, accuracy, f1, precision

# List to store performance metrics for each model
model_performance = []

# Random Forest Model Evaluation
print("Random Forest Performance:")
rf_pred, rf_accuracy, rf_f1, rf_precision = evaluate_model_performance(best_rf, X_test, y_test)
model_performance.append(('Random Forest', rf_accuracy, rf_f1, rf_precision))

# Support Vector Classifier Model Evaluation
print("\nSupport Vector Classifier Performance:")
svc_pred, svc_accuracy, svc_f1, svc_precision = evaluate_model_performance(best_svc, X_test, y_test)
model_performance.append(('SVC', svc_accuracy, svc_f1, svc_precision))

# K-Nearest Neighbors Model Evaluation
print("\nK-Nearest Neighbors Performance:")
knn_pred, knn_accuracy, knn_f1, knn_precision = evaluate_model_performance(best_knn, X_test, y_test)
model_performance.append(('KNN', knn_accuracy, knn_f1, knn_precision))

# RandomizedSearchCV Random Forest Model Evaluation
print("\nRandom Forest (RandomizedSearchCV) Performance:")
random_rf_pred, random_rf_accuracy, random_rf_f1, random_rf_precision = evaluate_model_performance(best_random_rf, X_test, y_test)
model_performance.append(('Randomized RF', random_rf_accuracy, random_rf_f1, random_rf_precision))

# Display a summary of performance metrics for all models
print("\nSummary of Model Performance:")
print(f"{'Model':<20}{'Accuracy':<10}{'F1 Score':<10}{'Precision':<10}")
for model_name, accuracy, f1, precision in model_performance:
    print(f"{model_name:<20}{accuracy:<10.4f}{f1:<10.4f}{precision:<10.4f}")

# Select the best model based on performance (e.g., highest F1 score)
best_model_name, best_model, best_model_pred = '', None, None
if rf_f1 >= svc_f1 and rf_f1 >= knn_f1 and rf_f1 >= random_rf_f1:
    best_model_name, best_model, best_model_pred = 'Random Forest', best_rf, rf_pred
elif svc_f1 >= rf_f1 and svc_f1 >= knn_f1 and svc_f1 >= random_rf_f1:
    best_model_name, best_model, best_model_pred = 'Support Vector Classifier', best_svc, svc_pred
elif knn_f1 >= rf_f1 and knn_f1 >= svc_f1 and knn_f1 >= random_rf_f1:
    best_model_name, best_model, best_model_pred = 'K-Nearest Neighbors', best_knn, knn_pred
else:
    best_model_name, best_model, best_model_pred = 'Random Forest (RandomizedSearchCV)', best_random_rf, random_rf_pred

print(f"\nBest Performing Model: {best_model_name}")

# Generate Confusion Matrix for the Best Model
conf_matrix = confusion_matrix(y_test, best_model_pred)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f'Confusion Matrix - {best_model_name}')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# Step 6: Stacked Model Performance Analysis
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Stacking Classifier --------------------
print("\nStacked Model Performance Analysis:")

# Combine Random Forest and Support Vector Classifier into a stacked model
estimators = [
    ('rf', best_rf),  # Random Forest from Step 4
    ('svc', best_svc)  # Support Vector Classifier from Step 4
]

stacked_model = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier(random_state=42))
stacked_model.fit(X_train, y_train)

# Evaluate Stacked Model Performance
y_pred_stacked = stacked_model.predict(X_test)

# Calculate accuracy, F1 score, and precision
stacked_accuracy = accuracy_score(y_test, y_pred_stacked)
stacked_f1 = f1_score(y_test, y_pred_stacked, average='weighted')
stacked_precision = precision_score(y_test, y_pred_stacked, average='weighted')

print(f"Stacked Model Accuracy: {stacked_accuracy:.4f}")
print(f"Stacked Model F1 Score: {stacked_f1:.4f}")
print(f"Stacked Model Precision: {stacked_precision:.4f}")

# Confusion Matrix for the Stacked Model
stacked_conf_matrix = confusion_matrix(y_test, y_pred_stacked)

# Plot the confusion matrix using seaborn heatmap
plt.figure(figsize=(8,6))
sns.heatmap(stacked_conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix - Stacked Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# -------------------- Analysis of Stacking Model Performance --------------------
# Compare the stacked model's performance to the individual models
best_accuracy = max(rf_accuracy, svc_accuracy, knn_accuracy, random_rf_accuracy)
if stacked_accuracy > best_accuracy:
    print("\nThe stacked model showed an improvement in accuracy compared to the individual models.")
    print("This improvement may be due to the complementary strengths of the Random Forest and SVC models.")
    print("Random Forest is robust for handling noisy data, while SVC excels in separating classes with clear margins.")
else:
    print("\nThe stacked model did not show significant improvement in accuracy.")
    print("This limited effectiveness might be due to the similarity in how the individual models handle the data.")
    print("Both models could be making similar predictions on the same instances, leading to minimal stacking benefits.")


# Step 7: Model Evaluation (Saving the model)
import joblib

# -------------------- Save the Selected Model --------------------
# Save the best model (from Step 5 or Step 6) to a file using Joblib
model_filename = 'best_stacked_model.joblib'  # You can change the name if using another model
joblib.dump(stacked_model, model_filename)
print(f"Model saved to {model_filename}")

# -------------------- Load the Model and Predict --------------------
# Load the saved model
loaded_model = joblib.load(model_filename)

# Define the new set of random coordinates for prediction
new_coordinates = [
    [9.375, 3.0625, 1.51],   # Random coordinate 1
    [6.995, 5.125, 0.3875],  # Random coordinate 2
    [0, 3.0625, 1.93],       # Random coordinate 3
    [9.4, 3, 1.8],           # Random coordinate 4
    [9.4, 3, 1.3]            # Random coordinate 5
]

# Predict the maintenance steps for the new coordinates using the loaded model
predicted_steps = loaded_model.predict(new_coordinates)

# Display the predictions
print("Predicted maintenance steps for the given coordinates:")
for coords, step in zip(new_coordinates, predicted_steps):
    print(f"Coordinates {coords} => Predicted Maintenance Step: {step}")

