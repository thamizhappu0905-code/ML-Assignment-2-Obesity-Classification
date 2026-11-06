import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title='Obesity Level Classification', page_icon='📊', layout='wide')

TARGET = 'NObeyesdad'
MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'K-Nearest Neighbors': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'Support Vector Machine': 'model/svm.joblib'
}
EXPECTED_FEATURES = [
    'Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight',
    'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF',
    'TUE', 'CALC', 'MTRANS'
]

st.title('Obesity Level Classification')
st.caption('Machine Learning Model Evaluation Dashboard')
st.write('Evaluate five trained multiclass classifiers using a labelled CSV test dataset. The application loads saved pipelines and does not retrain models.')

with st.sidebar:
    st.header('Evaluation Settings')
    selected_model = st.selectbox('Select classification model', list(MODEL_FILES.keys()), index=1)
    st.markdown('**Required target column:** `NObeyesdad`')
    st.markdown('**Required predictors:** 16 original UCI features')

uploaded_file = st.file_uploader('Upload labelled test dataset (CSV)', type=['csv'])

if uploaded_file is None:
    st.info('Upload the exported `test_data.csv` from the notebook to begin evaluation.')
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error('The uploaded file could not be read as a CSV.')
    st.exception(exc)
    st.stop()

missing_columns = [c for c in EXPECTED_FEATURES + [TARGET] if c not in data.columns]
if missing_columns:
    st.error('The uploaded CSV is missing required columns: ' + ', '.join(missing_columns))
    st.stop()

# Keep only the columns expected by the trained pipeline and the target.
X_test = data[EXPECTED_FEATURES].copy()
y_test = data[TARGET].copy()

model_path = MODEL_FILES[selected_model]
if not os.path.exists(model_path):
    st.error(f'Model file not found: {model_path}')
    st.stop()

try:
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_prob_raw = model.predict_proba(X_test)
except Exception as exc:
    st.error('The saved model could not be loaded or used for prediction. Ensure the deployment uses a compatible scikit-learn version.')
    st.exception(exc)
    st.stop()

classifier = model.named_steps['classifier']
model_classes = np.asarray(classifier.classes_)
class_order = np.asarray(sorted(np.unique(np.concatenate([np.asarray(y_test), model_classes]))))

# Align probability columns to the complete class order.
prob_df = pd.DataFrame(y_prob_raw, columns=model_classes)
missing_prob_classes = [c for c in class_order if c not in prob_df.columns]
if missing_prob_classes:
    st.error('The uploaded target contains class values not known to the trained model: ' + ', '.join(map(str, missing_prob_classes)))
    st.stop()
y_prob = prob_df[class_order].to_numpy()
y_true_bin = label_binarize(y_test, classes=class_order)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
mcc = matthews_corrcoef(y_test, y_pred)

try:
    auc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='weighted')
except ValueError:
    auc = np.nan

st.subheader(selected_model)
st.caption(f'Evaluated on {len(data):,} labelled observations')

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)
c1.metric('Accuracy', f'{accuracy:.4f}')
c2.metric('AUC (OvR, weighted)', 'N/A' if np.isnan(auc) else f'{auc:.4f}')
c3.metric('Precision (weighted)', f'{precision:.4f}')
c4.metric('Recall (weighted)', f'{recall:.4f}')
c5.metric('F1 Score (weighted)', f'{f1:.4f}')
c6.metric('MCC', f'{mcc:.4f}')

left, right = st.columns([1.05, 1])
with left:
    st.subheader('Confusion Matrix')
    cm = confusion_matrix(y_test, y_pred, labels=class_order)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_order, yticklabels=class_order)
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Actual Class')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    st.pyplot(fig)

with right:
    st.subheader('Classification Report')
    report = classification_report(y_test, y_pred, labels=class_order,
                                   output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    st.dataframe(report_df.round(4), use_container_width=True)

with st.expander('Preview uploaded test data'):
    st.dataframe(data.head(20), use_container_width=True)

with st.expander('Metric method'):
    st.markdown('''
- Accuracy: overall proportion of correct predictions.
- AUC: multiclass One-vs-Rest, weighted by class support.
- Precision, Recall and F1: weighted multiclass averages.
- MCC: multiclass Matthews Correlation Coefficient.
- All metrics are calculated from the uploaded labelled test data and predictions from the saved pipeline.
''')
