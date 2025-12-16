# %% [markdown]
# # **Project Title: Predicting divorce**
# by Kabir, Rodrigue and Sertac

# %% [markdown]
# **Research main objective**
#
# This project aims to develop and validate predictive models of divorce using multidimensional determinants (demographic, socioeconomic, relational, and psychological characteristics), and to identify the key factors that most strongly contribute to marital dissolution.
#
# Specifically, it seeks at
#
# -	Training and comparing multiple supervised learning models predicting divorce
# -	Identifiy the most important predictors of divorce
#
#
# **Research questions**
#
# -	Which supervised machine-learning model offers the most reliable and robust prediction of divorce?
# -	Which factors contribute most to predicting divorce ?
#
#
# **Methods**
# -	Exploration of the dataset
# -	Preparation of the dataset
# -	Training and comparison of supervised learning models
# -	Identification of  the most important predictors of divorce
#

# %%
# Load libraries that will be used throughout the project (will be continuouysly updated)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# %%
# Load the dataset
df = pd.read_csv('divorce_df.csv')
df.head()

# %%
# Make a copy of the original dataset
df1 = df.copy()
df1.head()

# %%
# Show the list of columns in the dataset
df1.columns

# %%
# Extract the dataset to be used for the project
df3 = df1.copy()
df3 = df1[['communication_score', 'financial_stress_level', 'mental_health_issues',
           'infidelity_occurred', 'social_support', 'domestic_violence_history', 'trust_score', 'divorced']]
df3.head()

# %%
# Produce summary statistics of the dataset
df3.describe()

# %%
# Generating frequency tables for binary variables
binary_columns = ['infidelity_occurred',
                  'domestic_violence_history', 'divorced', "mental_health_issues"]
for col in binary_columns:
    print(f"Frequency table for {col}:")
    print(df3[col].value_counts())
    print("\n")

# %%
# Check for missing values
df3.isnull().sum()

# %% [markdown]
# **Pre-processing and preparation of the dataset for machine learning**

# %%
# Make train-test split
# Define feature matrix X and target vector y
X = df3[['communication_score', 'financial_stress_level', 'mental_health_issues',
         'infidelity_occurred', 'social_support', 'domestic_violence_history', 'trust_score']]
y = df3['divorced']

# 2. Perform 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"--- Dataset Split Results ---")
print(f"Total samples: {len(df3)}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution:\n{y_train.value_counts(normalize=True)}")
print(f"y_test distribution:\n{y_test.value_counts(normalize=True)}")
print(f"-----------------------------")

# %%
# 2. Create preprocessing pipelines for numerical and categorical features
numerical_features = ["communication_score",
                      "financial_stress_level", "social_support", "trust_score"]
categorical_features = ["mental_health_issues",
                        "infidelity_occurred", "domestic_violence_history"]
numerical_transformer = StandardScaler()  # Scale continuous features
# Convert categorical features to numerical
categorical_transformer = OneHotEncoder(
    handle_unknown='ignore', sparse_output=False)

# 3. Combine transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# %% [markdown]
# **Training and comparison of supervised models**

# %%
# Define divorce_model1 (Logistic Regression)
divorce_model1 = Pipeline(steps=[(
    'preprocessor', preprocessor), ('classifier', LogisticRegression(random_state=1))])
# Train the model
divorce_model1.fit(X_train, y_train)
# Predict on the test set
y_pred = divorce_model1.predict(X_test)
# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Training Model (Logistic Regression Classifier) ---")

print("Making predictions for the following 5 couples:")
print(X.head())
print("The predictions are")
print(divorce_model1.predict(X.head()))

print("\n--- Model Evaluation Results---")
print(f"Accuracy on Test Set: {accuracy:.4f}")

print("\n--- Confusion Matrix ---")
print("    Predicted 0 | Predicted 1")
print(f"Actual 0: {conf_matrix[0][0]:>10} | {conf_matrix[0][1]:>10}")
print(f"Actual 1: {conf_matrix[1][0]:>10} | {conf_matrix[1][1]:>10}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%
# Define divorce_model1 (Random Forest Classifier)
divorce_model1 = Pipeline(steps=[(
    'preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=1))])
# Train the model
divorce_model1.fit(X_train, y_train)
# Predict on the test set
y_pred = divorce_model1.predict(X_test)
# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Training Model (Random Forest Classifier) ---")

print("Making predictions for the following 5 couples:")
print(X.head())
print("The predictions are")
print(divorce_model1.predict(X.head()))

print("\n--- Model Evaluation Results---")
print(f"Accuracy on Test Set: {accuracy:.4f}")

print("\n--- Confusion Matrix ---")
print("    Predicted 0 | Predicted 1")
print(f"Actual 0: {conf_matrix[0][0]:>10} | {conf_matrix[0][1]:>10}")
print(f"Actual 1: {conf_matrix[1][0]:>10} | {conf_matrix[1][1]:>10}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# %%
# Define divorce_model1 (Xgboost Classifier)
divorce_model1 = Pipeline(steps=[(
    'preprocessor', preprocessor), ('classifier', XGBClassifier(random_state=1))])
# Train the model
divorce_model1.fit(X_train, y_train)
# Predict on the test set
y_pred = divorce_model1.predict(X_test)
# Evaluation of the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n--- Training Model (XGBoost Classifier) ---")

print("Making predictions for the following 5 couples:")
print(X.head())
print("The predictions are")
print(divorce_model1.predict(X.head()))

print("\n--- Model Evaluation Results---")
print(f"Accuracy on Test Set: {accuracy:.4f}")

print("\n--- Confusion Matrix ---")
print("    Predicted 0 | Predicted 1")
print(f"Actual 0: {conf_matrix[0][0]:>10} | {conf_matrix[0][1]:>10}")
print(f"Actual 1: {conf_matrix[1][0]:>10} | {conf_matrix[1][1]:>10}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
