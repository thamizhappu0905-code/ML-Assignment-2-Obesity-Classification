Obesity Level Classification using Machine Learning
1. Problem Statement
The objective of this project is to develop and compare multiple machine learning classification models for predicting obesity levels based on eating habits, physical condition, and lifestyle characteristics.

The following five classification algorithms are implemented and evaluated:

Logistic Regression
Decision Tree
K-Nearest Neighbors
Naive Bayes
Support Vector Machine
The models are evaluated using the following performance metrics:

Accuracy
AUC
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)
The final trained models are also integrated into an interactive Streamlit application for model evaluation using uploaded test data.
2. Dataset Description
Dataset: Estimation of Obesity Levels Based on Eating Habits and Physical Condition

Source: UCI Machine Learning Repository

The original dataset contains:

2,111 observations
16 predictor variables
1 target variable
17 columns in total
The target variable is:

NObeyesdad

The target represents seven weight or obesity-level categories:

Insufficient Weight
Normal Weight
Overweight Level I
Overweight Level II
Obesity Type I
Obesity Type II
Obesity Type III
Therefore, the machine learning problem is treated as a multiclass classification problem.

3. Predictor Variables
The dataset contains a combination of numerical and categorical predictors representing physical characteristics, eating habits, lifestyle behaviour, and transportation patterns.

Numerical features
Examples include:

Age
Height
Weight
Frequency of vegetable consumption (FCVC)
Number of main meals (NCP)
Daily water consumption (CH2O)
Physical activity frequency (FAF)
Technology usage time (TUE)
Categorical features
Examples include:

Gender
Family history of overweight
Frequent consumption of high-calorie food (FAVC)
Food consumption between meals (CAEC)
Smoking (SMOKE)
Calorie consumption monitoring (SCC)
Alcohol consumption (CALC)
Transportation mode (MTRANS)
4. Data Understanding and Cleaning
Before building the machine learning models, the dataset was examined to understand its structure and quality.

The following checks were performed:

Dataset dimensions were inspected.
Column names and data types were verified.
Numerical variables were statistically summarized.
Categorical variables were reviewed.
Missing values were checked.
Exact duplicate observations were identified.
Duplicate observations were removed before train-test splitting.
Target-class distribution was analyzed.
The original source dataset was not modified.

A cleaned working copy was used for model development.

5. Exploratory Data Analysis
Exploratory Data Analysis was performed before model development.

The analysis included:

Distribution of target classes
Age distribution
Height distribution
Weight distribution
Weight variation across obesity categories
Correlation analysis of numerical variables
The purpose of EDA was to understand the structure of the dataset and identify patterns before applying machine learning algorithms.

6. Data Preprocessing
The dataset contains both numerical and categorical predictor variables. Different preprocessing methods were therefore applied to each feature type.

Numerical variables
Numerical variables were standardized using:

StandardScaler

Standardization places numerical predictors on comparable scales, which is particularly important for distance-based and margin-based algorithms such as K-Nearest Neighbors and Support Vector Machine.

Categorical variables
Categorical variables were transformed using:

OneHotEncoder

The encoder was configured to safely handle categories that may appear during prediction.

Combined preprocessing
A scikit-learn ColumnTransformer was used to combine numerical and categorical preprocessing.

The preprocessing transformer was incorporated directly within each machine learning pipeline.

This ensures that preprocessing is fitted only using the training dataset rather than the complete dataset, thereby reducing the risk of data leakage.

7. Train-Test Split
The cleaned dataset was split into training and testing samples using an:

80% training / 20% testing split

A stratified split was used so that the distribution of the seven obesity classes was maintained across both training and testing datasets.

Where applicable:

random_state = 42

was used to support reproducibility.

The final held-out test dataset contains:

418 labelled observations

The same test dataset was used to evaluate all five classification algorithms.

8. Machine Learning Models
Five classification algorithms were implemented.

8.1 Logistic Regression
Logistic Regression was used as a linear multiclass classification baseline.

The algorithm estimates class probabilities and provides a useful benchmark for comparing more complex nonlinear algorithms.

8.2 Decision Tree
Decision Tree is a nonlinear classification algorithm that recursively divides feature space using decision rules.

The model can capture complex relationships and interactions between predictors without requiring a linear decision boundary.

8.3 K-Nearest Neighbors
K-Nearest Neighbors classifies an observation using the classes of nearby observations in the transformed feature space.

Since KNN relies on distance calculations, feature scaling is particularly important.

8.4 Naive Bayes
Gaussian Naive Bayes was implemented as a probabilistic classification algorithm.

The algorithm assumes conditional independence between predictor variables given the target class.

8.5 Support Vector Machine
Support Vector Machine was implemented using an RBF kernel.

Probability estimation was enabled so that multiclass AUC could also be calculated.

SVM is capable of modelling nonlinear decision boundaries using the kernel approach.

9. Evaluation Metrics
Each machine learning model was evaluated using six performance measures.

Accuracy
Measures the proportion of test observations that were classified correctly.

AUC
Measures the ability of the model to distinguish between classes.

Since this is a multiclass classification problem, AUC was calculated using a weighted One-vs-Rest approach.

Precision
Measures how reliable the predicted classes are.

Weighted multiclass averaging was used.

Recall
Measures how successfully actual observations from the different classes were identified.

Weighted multiclass averaging was used.

F1 Score
Represents the harmonic balance between Precision and Recall.

Weighted multiclass averaging was used.

Matthews Correlation Coefficient
MCC evaluates agreement between predicted and actual classes while considering the complete confusion matrix.

MCC was calculated directly from multiclass predictions.

10. Model Comparison
The final model evaluation results obtained from the held-out test dataset are shown below.

Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Decision Tree	0.9330	0.9610	0.9340	0.9330	0.9326	0.9220
Support Vector Machine	0.9115	0.9940	0.9121	0.9115	0.9114	0.8967
Logistic Regression	0.8971	0.9861	0.8950	0.8971	0.8954	0.8800
K-Nearest Neighbors	0.8278	0.9662	0.8204	0.8278	0.8136	0.8017
Naive Bayes	0.5144	0.8291	0.5252	0.5144	0.4650	0.4566
11. Model Performance Observations
Decision Tree
Decision Tree produced the strongest overall classification performance.

The model achieved:

Accuracy: 0.9330
AUC: 0.9610
Precision: 0.9340
Recall: 0.9330
F1 Score: 0.9326
MCC: 0.9220
Decision Tree achieved the highest Accuracy, Precision, Recall, F1 Score, and MCC among the five models.

The MCC value of 0.9220 indicates strong overall agreement between predicted and actual obesity classes.

Support Vector Machine
Support Vector Machine also demonstrated strong classification performance.

The model achieved:

Accuracy: 0.9115
AUC: 0.9940
Precision: 0.9121
Recall: 0.9115
F1 Score: 0.9114
MCC: 0.8967
SVM achieved the highest AUC of all five algorithms, indicating excellent ability to discriminate between the different obesity classes using the multiclass One-vs-Rest evaluation method.

However, its Accuracy, F1 Score, and MCC were slightly lower than those of Decision Tree.

Logistic Regression
Logistic Regression provided a strong linear baseline.

The model achieved:

Accuracy: 0.8971
AUC: 0.9861
Precision: 0.8950
Recall: 0.8971
F1 Score: 0.8954
MCC: 0.8800
The relatively strong performance indicates that a substantial amount of the class structure can be captured through a linear multiclass decision mechanism after appropriate preprocessing.

K-Nearest Neighbors
KNN produced moderate classification performance.

The model achieved:

Accuracy: 0.8278
AUC: 0.9662
Precision: 0.8204
Recall: 0.8278
F1 Score: 0.8136
MCC: 0.8017
Although KNN achieved a relatively high AUC, its Accuracy, F1 Score, and MCC were below Decision Tree, SVM, and Logistic Regression.

Naive Bayes
Naive Bayes produced the weakest overall classification results.

The model achieved:

Accuracy: 0.5144
AUC: 0.8291
Precision: 0.5252
Recall: 0.5144
F1 Score: 0.4650
MCC: 0.4566
The lower performance suggests that the conditional-independence assumptions of Gaussian Naive Bayes are less suitable for the relationships represented by this dataset.

12. Best Performing Model
Based on overall classification performance, Decision Tree was selected as the best-performing model.

Decision Tree achieved:

Metric	Score
Accuracy	0.9330
AUC	0.9610
Precision	0.9340
Recall	0.9330
F1 Score	0.9326
MCC	0.9220
Although Support Vector Machine achieved a higher AUC of 0.9940, Decision Tree provided the strongest overall combination of:

Accuracy
Precision
Recall
F1 Score
MCC
Therefore, Decision Tree is considered the strongest overall classifier in this experiment.

13. Model Persistence
Each trained classifier was saved using Joblib.

The following trained model files are included:

model/
├── decision_tree.joblib
├── knn.joblib
├── logistic_regression.joblib
├── naive_bayes.joblib
└── svm.joblib
