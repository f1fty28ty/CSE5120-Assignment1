#Step 1:
# Import libraries
# In this section, you can use a search engine to look for the functions that will help you implement the following steps
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import category_encoders as ce


#Step 2:
# Load dataset and show basic statistics
# 1. Show dataset size (dimensions)
# 2. Show what column names exist for the 49 attributes in the dataset
# 3. Show the distribution of the target class CES 4.0 Percentile Range column
# 4. Show the percentage distribution of the target class CES 4.0 Percentile Range column
df = pd.read_csv('disadvantaged_communities.csv')

print("Dataset Size (rows, columns):", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nTarget Class Distribution (CES 4.0 Percentile Range):")
print(df['CES 4.0 Percentile Range'].value_counts())

print("\nTarget Class Percentage Distribution:")
print(df['CES 4.0 Percentile Range'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')


# Step 3:
#Clean the dataset - you can eitherhandle the missing values in the dataset
# with the mean of the columns attributes or remove rows the have missing values.
target_col = 'CES 4.0 Percentile Range'
df = df.dropna(subset=[target_col])

# Drop columns with 20%+ missing values as advised
high_missing_cols = [
    'Linguistic Isolation Pctl', 'Unemployment', 'Lead', 'Traffic',
    'Low Birth Weight', 'Education', 'Housing Burden'
]
df = df.drop(columns=[c for c in high_missing_cols if c in df.columns])

# Fill remaining missing numeric values with column mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill remaining missing string/object values with column mode
object_cols = df.select_dtypes(include=['object']).columns
for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


# Step 4: 
#Encode the Categorical Variables - Using OrdinalEncoder from the category_encoders library to encode categorical variables as ordinal integers
# Note: target column CES 4.0 Percentile Range is NOT encoded
categorical_features = [c for c in df.select_dtypes(include=['object', 'str']).columns if c != target_col]
encoder = ce.OrdinalEncoder(cols=categorical_features)
df = encoder.fit_transform(df)

# Fill any remaining NaNs after encoding
df = df.fillna(df.mean(numeric_only=True))


# Step 5: 
# Separate predictor variables from the target variable (attributes (X) and target variable (y) as we did in the class)
# Create train and test splits for model development. Use the 90% and 20% split ratio
# Use stratifying (stratify=y) to ensure class balance in train/test splits
# Name them as X_train, X_test, y_train, and y_test
# Name them as X_train, X_test, y_train, and y_test
drop_cols = [c for c in ['CES 4.0 Score', 'CES 4.0 Percentile', 'DAC category', 'Census Tract'] if c in df.columns]
X = df.drop(columns=[target_col] + drop_cols)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# Keep unscaled copies for Random Forest (Step 9)
X_train_rf = X_train.copy()
X_test_rf  = X_test.copy()


# Do not do steps 6 - 8 for the Ramdom Forest Model
# Step 6:
# Standardize the features (Import StandardScaler here)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# Step 7:
# Below is the code to convert X_train and X_test into data frames for the next steps
cols = X_train.columns
X_train = pd.DataFrame(X_train_scaled, columns=cols) # pd is the imported pandas lirary - Import pandas as pd
X_test = pd.DataFrame(X_test_scaled, columns=cols) # pd is the imported pandas lirary - Import pandas as pd


# Step 8 - Build and train the SVM classifier
# Train SVM with the following parameters. (use the parameters with the highest accuracy for the model)
   #    1. RBF kernel
   #    2. C=10.0 (Higher value of C means fewer outliers)
   #    3. gamma = 0.3 (Linear)
svm_model = SVC(kernel='rbf', C=8.0, gamma=0.3, random_state=42)
svm_model.fit(X_train, y_train)

# Test the above developed SVC on unseen pulsar dataset samples
y_pred_svm = svm_model.predict(X_test)

# compute and print accuracy score
print('\nSVM Accuracy: {0:0.4f}'.format(accuracy_score(y_test, y_pred_svm)))

# Save your SVC model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
with open('SvmClassifier.sav', 'wb') as f:
    pickle.dump(svm_model, f)


# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred_svm)
print('\nSVM Confusion Matrix:')
print(cm)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
i = np.argmax(np.diag(cm))  # use the class with the most correct predictions
TP = cm[i,i]
FP = cm[:,i].sum() - TP
FN = cm[i,:].sum() - TP
TN = cm.sum() - TP - FP - FN

# Compute Precision and use the following line to print it
precision = TP / (TP + FP)
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN)
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP)
print('Specificity : {0:0.3f}'.format(specificity))


# Step 9: Build and train the Random Forest classifier
# Train Random Forest  with the following parameters.
# (n_estimators=10, random_state=0)
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
rf_model.fit(X_train_rf, y_train)

# Test the above developed Random Forest model on unseen DACs dataset samples
y_pred_rf = rf_model.predict(X_test_rf)

# compute and print accuracy score
print('\nRandom Forest Accuracy: {0:0.4f}'.format(accuracy_score(y_test, y_pred_rf)))

# Save your Random Forest model (whatever name you have given your model) as .sav to upload with your submission
# You can use the library pickle to save and load your model for this assignment
with open('RfClassifier.sav', 'wb') as f:
    pickle.dump(rf_model, f)

# Optional: You can print test results of your model here if you want. Otherwise implement them in evaluation.py file
# Get and print confusion matrix
cm = confusion_matrix(y_test, y_pred_rf)
print('\nRandom Forest Confusion Matrix:')
print(cm)

# Below are the metrics for computing classification accuracy, precision, recall and specificity
i = np.argmax(np.diag(cm))  # use the class with the most correct predictions
TP = cm[i,i]
FP = cm[:,i].sum() - TP
FN = cm[i,:].sum() - TP
TN = cm.sum() - TP - FP - FN

# Compute Classification Accuracy and use the following line to print it
classification_accuracy = accuracy_score(y_test, y_pred_rf)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))

# Compute Precision and use the following line to print it
precision = TP / (TP + FP)
print('Precision : {0:0.3f}'.format(precision))

# Compute Recall and use the following line to print it
recall = TP / (TP + FN)
print('Recall or Sensitivity : {0:0.3f}'.format(recall))

# Compute Specificity and use the following line to print it
specificity = TN / (TN + FP)
print('Specificity : {0:0.3f}'.format(specificity))