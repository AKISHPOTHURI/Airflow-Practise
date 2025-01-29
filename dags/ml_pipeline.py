from airflow.models import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn import metrics
import numpy as np
import pickle

# Function to prepare data
def prepare_data():
    print("----------Started Data Preparation---------------")
    df = pd.read_csv("https://raw.githubusercontent.com/TripathiAshutosh/MLOps-IRIS/refs/heads/master/data/Iris.csv")
    df = df.dropna()
    df.to_csv('final_df.csv', index=False)
    print("----------Data Saved Successfully---------------")

# Function to split data into train and test sets
def split_train_test():
    print("------------split_train_test Started-------------")
    final_data = pd.read_csv('final_df.csv')
    target_column = 'Species'
    
    # Ensure y is a Series, not a DataFrame
    X = final_data.drop(columns=[target_column])  # Features
    y = final_data[target_column]  # Target variable as a Series

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    print("-------Data Saved Successfully---------")

# Function to train a logistic regression model
def training_basic_classifier():
    print("--------Training Basic Classifier Started---------")
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)

    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    print("Logistic Regression Classifier Trained Successfully")

# Function to predict on test data
def predict_on_test_data():
    print("-----Prediction on Test Started--------")
    with open('model.pkl', 'rb') as f:
        logistic_reg_model = pickle.load(f)

    X_test = np.load('X_test.npy', allow_pickle=True)
    y_pred = logistic_reg_model.predict(X_test)

    np.save('y_pred.npy', y_pred)
    print("----------Prediction on Test Completed-------")

# Function to predict probability on test data
def predict_prob_on_test_data():
    print("---------Predicting Probability on Test Data----------")
    with open('model.pkl', 'rb') as f:
        logistic_reg_model = pickle.load(f)

    X_test = np.load('X_test.npy', allow_pickle=True)
    y_pred_prob = logistic_reg_model.predict_proba(X_test)

    np.save('y_pred_prob.npy', y_pred_prob)
    print("----------Probability Prediction Completed-------")

# Function to calculate and display metrics
def get_metrics():
    print("----------Get Metrics Started-----------")
    y_test = np.load('y_test.npy', allow_pickle=True)
    y_pred = np.load('y_pred.npy', allow_pickle=True)
    y_pred_prob = np.load('y_pred_prob.npy', allow_pickle=True)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')  # Multi-class fix
    recall = recall_score(y_test, y_pred, average='weighted')  # Multi-class fix
    entropy = log_loss(y_test, y_pred_prob)  

    print(metrics.classification_report(y_test, y_pred))
    print("\n Model Metrics:", {
        'accuracy': round(acc, 2),
        'precision': round(prec, 2),
        'recall': round(recall, 2),
        'entropy': round(entropy, 2)
    })

# Define Airflow DAG
with DAG(
    dag_id='ml_pipeline_demo',
    schedule='@daily',  # Replace schedule_interval with schedule
    start_date=datetime(2025, 1, 29),  
    catchup=False
) as dag:
    
    task_prepare_data = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    task_split_train_test = PythonOperator(
        task_id='split_train_test',
        python_callable=split_train_test
    )

    task_training_basic_classifier = PythonOperator(
        task_id='training_basic_classifier',
        python_callable=training_basic_classifier
    )

    task_predict_on_test_data = PythonOperator(
        task_id='predict_on_test_data',
        python_callable=predict_on_test_data
    )

    task_predict_prob_on_test_data = PythonOperator(
        task_id='predict_prob_on_test_data',
        python_callable=predict_prob_on_test_data
    )

    task_get_metrics = PythonOperator(
        task_id='get_metrics',
        python_callable=get_metrics
    )

    # Task Execution Flow
    task_prepare_data >> task_split_train_test >> task_training_basic_classifier >> task_predict_on_test_data >> task_predict_prob_on_test_data >> task_get_metrics
