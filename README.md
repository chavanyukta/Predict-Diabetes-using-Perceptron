# Diabetes Prediction Using Perceptron Algorithm

## Project Overview
This project implements a Perceptron algorithm to predict diabetes outcomes using the Pima Indians Diabetes Database from the UCI Machine Learning Repository. The Perceptron is a simple linear classifier inspired by biological neurons and is used here for binary classification to identify diabetes presence based on health-related features.
The project covers data loading, preprocessing, model training with hyperparameter tuning, and performance evaluation using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC metrics.

## Dataset
- Source: Pima Indians Diabetes Database (UCI Machine Learning Repository)
- Features include: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target variable: Outcome (0 = no diabetes, 1 = diabetes)
- Data format: Text file with labeled feature-value pairs (LIBSVM format)

## Requirements
- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn

## Selected Columns for Project
- All eight health-related attributes mentioned above are used as input features.

## Project Structure
─ perceptron_diabetes.py # Main script for training and evaluating the Perceptron model
─ data_preprocessing.py # Module for loading and preprocessing the dataset
─ model_training.py # Module implementing the Perceptron training and hyperparameter tuning
─ result_visualization.py # Module for plotting confusion matrix, ROC curve, and classification reports
─ data/ # Directory containing dataset files
─ README.md # Project documentation (this file)
─ requirements.txt # Required Python libraries

## Project Report
The full project report detailing the methodology, results, and analysis is available in the file:  
[Predict Diabetes using Perceptron Report.pdf](./Predict%20Diabetes%20using%20Perceptron%20Report.pdf)

## Contact
Yukta Sanjiv Chavan
Email: chavanyukta@gmail.com

