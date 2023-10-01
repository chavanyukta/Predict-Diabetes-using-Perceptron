#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Load and preprocess the data
import numpy as np

# Define a function to load the data from your short data version.
def load_data():
    data = []
    labels = []
    with open("C:/Users/chava/Downloads/csie.ntu.edu.tw_~cjlin_libsvmtools_datasets_binary_diabetes.txt", 'r') as file:
        for line in file:
            parts = line.strip().split()
            label = int(parts[0])
            features = {}
            for part in parts[1:]:
                index, value = part.split(':')
                features[int(index)] = float(value)
            data.append(features)
            labels.append(label)
    return np.array(data), np.array(labels)

# Load the data
data, labels = load_data()
print(data)


# In[32]:


# Split the data

from sklearn.model_selection import train_test_split

X_training, X_testing, y_training, y_testing = train_test_split(data, labels, test_size=0.3, random_state=50)
import numpy as np


# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
    'max_iter': [100, 1000, 10000],  # Increase the max_iter
}

grid_search = GridSearchCV(Perceptron(), param_grid, cv=5)
grid_search.fit(X_training_array, y_training)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

best_model = grid_search.best_estimator_

# Convert the dictionary-based data to a NumPy array
X_training_array = np.array([list(sample.values()) for sample in X_training])
# Convert the dictionary-based testing data to a NumPy array
X_testing_array = np.array([list(sample.values()) for sample in X_testing])


# In[33]:


# Implementing perceptron algorithm
from sklearn.linear_model import Perceptron

perceptron = Perceptron(alpha=0.0001, max_iter=100)
perceptron.fit(X_training_array, y_training)


# In[34]:


# Train the Perceptron model on the training data
perceptron.fit(X_training_array, y_training)

# Make predictions on the testing data
y_prediction = perceptron.predict(X_testing_array)

# Evaluate the model's performance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_testing, y_prediction)
precision = precision_score(y_testing, y_prediction)
recall = recall_score(y_testing, y_prediction)
f1 = f1_score(y_testing, y_prediction)

print("Perceptron Model Performance:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


# In[38]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
# Confusion matrix
cm = confusion_matrix(y_testing, y_prediction)
sns.heatmap(cm, annot=True, fmt="d")

print("Confusion Matrix:\n", cm)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
# Report
report = classification_report(y_testing, y_prediction)
print("Classification Report:\n", report)


# In[21]:


# ROC curve
from sklearn.metrics import roc_curve, roc_auc_score

y_scores = perceptron.decision_function(X_testing_array)
fpr, tpr, _ = roc_curve(y_testing, y_scores)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()

# You can also calculate the AUC (Area Under the Curve)
auc = roc_auc_score(y_testing, y_scores)
print(f"AUC: {auc}")


# In[ ]:




