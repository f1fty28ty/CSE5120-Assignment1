# 1. import any required library to load dataset, open files (os), print confusion matrix and accuracy score
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import category_encoders as ce


# 2. Create test set if you like to do the split programmatically or if you have not already split the data at this point
df = pd.read_csv('disadvantaged_communities.csv')

target_col = 'CES 4.0 Percentile Range'
df = df.dropna(subset=[target_col])

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

categorical_features = [c for c in df.select_dtypes(include=['object', 'str']).columns if c != target_col]
encoder = ce.OrdinalEncoder(cols=categorical_features)
df = encoder.fit_transform(df)

df = df.fillna(df.mean(numeric_only=True))

drop_cols = [c for c in ['CES 4.0 Score', 'CES 4.0 Percentile', 'DAC category', 'Census Tract'] if c in df.columns]
X = df.drop(columns=[target_col] + drop_cols)
y = df[target_col]

# 90/10 split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)

# Scale for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
cols = X_train.columns
X_test_svm = pd.DataFrame(X_test_scaled, columns=cols)


# 3. Load your saved model for dissadvantaged communities classification 
#that you saved in dissadvantaged_communities_classification.py via Pikcle
with open('SvmClassifier.sav', 'rb') as f:
    svm_model = pickle.load(f)

with open('RfClassifier.sav', 'rb') as f:
    rf_model = pickle.load(f)


# 4. Make predictions on test_set created from step 2
y_pred_svm = svm_model.predict(X_test_svm)
y_pred_rf  = rf_model.predict(X_test)


# 5. use predictions and test_set (X_test) classifications to print the following:
#    1. confution matrix, 2. accuracy score, 3. precision, 4. recall, 5. specificity
#    You can easily find the formulae for Precision, Recall, and Specificity online.

print("=" * 50)
print("SVM Evaluation")
print("=" * 50)

# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
print('Confusion Matrix:')
print(cm)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
i = np.argmax(np.diag(cm))
TP = cm[i,i]
FP = cm[:,i].sum() - TP
FN = cm[i,:].sum() - TP
TN = cm.sum() - TP - FP - FN

print('Accuracy    : {0:0.4f}'.format(accuracy_score(y_test, y_pred_svm)))
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
specificity = TN / (TN + FP)
print('Precision   : {0:0.3f}'.format(precision))
print('Recall      : {0:0.3f}'.format(recall))
print('Specificity : {0:0.3f}'.format(specificity))

print("\n" + "=" * 50)
print("Random Forest Evaluation")
print("=" * 50)

# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
print('Confusion Matrix:')
print(cm)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
i = np.argmax(np.diag(cm))
TP = cm[i,i]
FP = cm[:,i].sum() - TP
FN = cm[i,:].sum() - TP
TN = cm.sum() - TP - FP - FN

print('Accuracy    : {0:0.4f}'.format(accuracy_score(y_test, y_pred_rf)))
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
specificity = TN / (TN + FP)
print('Precision   : {0:0.3f}'.format(precision))
print('Recall      : {0:0.3f}'.format(recall))
print('Specificity : {0:0.3f}'.format(specificity))