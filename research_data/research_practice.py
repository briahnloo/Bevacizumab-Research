import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import os
import sys


np.random.seed(42)

# Change the working directory to the location of the script
os.chdir(os.path.dirname(os.path.realpath(sys.argv[0])))

# read files
bevacizumab = pd.read_csv('bevacizumab_dataset.csv')
patient_list = pd.read_csv('final_patient_list.csv')

# merge the datasets
df = pd.merge(bevacizumab, patient_list, on = ['Patient ID'])

# convert string in Image No. to float
df = df.drop('Image No.', axis=1)

# Replace values where 'CA-125 before' > 200 with 201
df['CA-125 before'] = df['CA-125 before'].replace(np.nan, 201)

# test how many common IDs between bevacizumab and patient_list
common_ids = set(bevacizumab['Patient ID']).intersection(set(patient_list['Patient ID']))
print(f'Number of common patient IDs: {len(common_ids)}')

# Preprocessing
# drop duplicate patient IDs
df = df.drop_duplicates(subset = ['Patient ID'])
df = df.drop('No._x', axis=1)
df = df.drop('No._y', axis=1)

# find unique values in the 'operations' column
print(df['operation'].unique())

# Convert all operation, Diagnosis, method for avastin use, FIGO stage to lowercase
df['operation'] = df['operation'].str.lower()
df['Diagnosis'] = df['Diagnosis'].str.lower()
df['method for avastin use'] = df['method for avastin use'].str.lower()
df['FIGO stage'] = df['FIGO stage'].str.lower()

# One-hot encoding
# now, df will have new columns 'FIGO stage_I', 'FIGO stage_II', 'FIGO stage_III' and 'FIGO stage_IV'
# with values 0 or 1 indicating whether the patient is in that stage or not.
df = pd.get_dummies(df, columns=['operation', 'Diagnosis', 'method for avastin use', 'FIGO stage'], drop_first=True)

df['CA-125 before'] = df['CA-125 before'].replace('H>200', 201)
df['CA-125 before'] = df['CA-125 before'].replace('H> 200', 201)
df['CA-125 after'] = df['CA-125 after'].replace('H>200', 201)

# After replacing, convert them to float
df['CA-125 before'] = df['CA-125 before'].astype(float)
df['CA-125 after'] = df['CA-125 after'].astype(float)


# subtract start date from end date to get avastin usage duration
df['starting date for use of avastin '] = pd.to_datetime(df['starting date for use of avastin '])
df['End date for use of avastin'] = pd.to_datetime(df['End date for use of avastin'])

# make the new column
df['duration of avastin use'] = (df['End date for use of avastin'] - df['starting date for use of avastin ']).dt.days
print(df['duration of avastin use'])

# strip spaces in column names
df.columns = df.columns.str.strip()

# strip spaces in cell values
df = df.applymap(lambda x: x.strip() if type(x)==str else x)



# save cleaned data


# # create histogram FIGURE 1!!!
# sns.histplot(df ['Treatment effect' ])
# plt.xlabel('treatment' )
# plt.ylabel( 'Count')
# plt.title('Histogram of treatment')
# plt.show()


# # Visualize the distribution of 'CA-125 before' and 'CA-125 after' treatments FIGURE 2
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.hist(df['CA-125 before'].dropna(), bins=30)
# plt.title('Distribution of CA-125 before treatment')

# plt.subplot(1, 2, 2)
# plt.hist(df['CA-125 after'].dropna(), bins=30)
# plt.title('Distribution of CA-125 after treatment')

# plt.show()


# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.hist(df['CA-125 before'].dropna(), bins=30)
# plt.title('Distribution of CA-125 before treatment')

# plt.subplot(1, 2, 2)
# plt.hist(df['CA-125 after'].dropna(), bins=30)
# plt.title('Distribution of CA-125 after treatment')
# plt.show()

# # Analyze the relationship between 'CA-125 before' and 'CA-125 after' treatments FIGURE 3
# plt.scatter(df['CA-125 before'], df['CA-125 after'])
# plt.xlabel('CA-125 before treatment')
# plt.ylabel('CA-125 after treatment')
# plt.title('Relationship between CA-125 before and after treatments')
# plt.show()
# #histogramn with the distribution of age
# plt.hist(df['Age'], bins=10, alpha=0.5)
# plt.title('Distribution of Age')
# plt.show()

# # plot relationship between BMI and CA-125 before variable
# sns.scatterplot(x='BMI', y='CA-125 before', data=df)
# plt.title('BMI vs CA-125 before')
# plt.show()

# # plot relationship between BMI and CA-125 after variable
# sns.scatterplot(x='BMI', y='CA-125 after', data=df)
# plt.title('BMI vs CA-125 after')
# plt.show()

# # box plot of CA-125 levels before and after treatment
# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# sns.boxplot(x='Treatment effect', y='CA-125 before', data=df)
# plt.title('Boxplot of CA-125 before treatment')

# plt.subplot(1, 2, 2)
# sns.boxplot(x='Treatment effect', y='CA-125 after', data=df)
# plt.title('Boxplot of CA-125 after treatment')
# plt.show()

# # heatmat correlation
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Heatmap')
# plt.show()




# graph age bmi etc
# do long transform to edit distribution; less skew
# binning
# normalize features before split
# paste categorical variables, string in directly, everything floats by the end
#   FIGO stage 1-4, one hot encoding transform into 1,2,3,4
# anaconda app, search for word conda within computer


# # create a boxplot for 'duration of avastin use' grouped by 'Treatment effect'
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df, x='Treatment effect', y='duration of avastin use')
# plt.title('Boxplot of Duration of Avastin Use Grouped by Treatment Effect')
# plt.show()

# # create a scatter plot for 'duration of avastin use' vs 'Age'
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='Age', y='duration of avastin use')
# plt.title('Scatterplot of Duration of Avastin Use vs Age')
# plt.show()





# drop 'Unnamed: 5' column until i find out what it is
df.drop('Unnamed: 5', axis=1, inplace=True)


# Numerically encode the 'Treatment effect' column
df['Treatment effect'] = df['Treatment effect'].str.lower().map({'effective': 1, 'invalid': 0})



df = df.drop('operation date', axis=1)
df = df.drop('recurrent date',axis=1)
df = df.drop('Date of death',axis=1)
df = df.drop('duration of avastin use', axis =1)
df = df.drop('starting date for use of avastin',axis=1)
df = df.drop('End date for use of avastin',axis=1)

print(df.dtypes)

# df = df.drop(['starting date for use of avastin', 'End date for use of avastin'], axis=1)

df = df.drop('Patient ID', axis=1)
# Continue with the feature selection
X = df.drop('Treatment effect', axis=1)

y = df['Treatment effect']




# Split the data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train)

# Assuming 'df' is your DataFrame
# for col in df.columns:
#     if df[col].astype(str).str.contains('7/18/08').any():
#         print(f"'7/18/08' found in column: {col}")




print('the types of datafiles that exist', df.dtypes)


selected_features = ['CA-125 before', 'CA-125 after', 'Age',
                     'number of avastin administration', 'BMI',
                     'operation_optimal debulking', 'operation_suboptimal debulking',
                     'Diagnosis_emac', 'method for avastin use_front line->maintenance',
                     'FIGO stage_iv']

# Select only the columns in selected_features
df_selected = df[selected_features]

# Print the first 5 rows of the new DataFrame
print(df_selected.head())

features = X_train
target = y_train
features = features.select_dtypes(exclude=['datetime64'])

# perform feature selection using chi-squared test to get 10 best features
selector = SelectKBest (chi2, k=10)
selector.fit (features, target)
selected_features = features.columns[selector.get_support(indices=True)]
print("Selected features:", selected_features)


# # reduce our feature sets to only include these selected features.
# X_train = X_train[selected_features]
# X_val = X_val[selected_features]
# X_test = X_test[selected_features]


df.to_csv('combined_data2.csv', index=False)



# Perform model creation and cross-validation
model = LogisticRegression()

print(X_train.shape)

# perform cross-validation
# define the method
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=1)

# compute cross-validation scores
cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

# print out cross validation scores and their averages
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores)*100:.2f}%")


# train the model using the training set
model.fit(X_train, y_train)

# use the trained model to predict the target variable in the validation set
y_pred_val = model.predict(X_val)





# calculate the accuracy of the model on validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
print(f"Validation Accuracy: {accuracy_val*100:.2f}%")

# confusion matrix and classification report for validation set
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))
print("\nValidation Classification Report:")
print(classification_report(y_val, y_pred_val))



# predictions;use the trained model to predict the target variable in the test set
y_pred_test = model.predict(X_test)

# calculate the accuracy of the model on test set
accuracy_test = accuracy_score(y_test, y_pred_test)
print(f"\nTest Accuracy: {accuracy_test*100:.2f}%")

# confusion matrix and classification report for test set
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test))


# initialize StandardScaler
scaler = StandardScaler()

# fit on your training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# add Regularization to logistic regression model
# L1 regularization
model_l1 = LogisticRegression(max_iter=1000, penalty='l1', solver='liblinear')
model_l1.fit(X_train_scaled, y_train)
y_pred_l1 = model_l1.predict(X_test_scaled)
accuracy_l1 = accuracy_score(y_test, y_pred_l1)
print(f"L1 Regularized Logistic Regression Validation Accuracy: {accuracy_l1*100:.2f}%")

# L2 regularization
model_l2 = LogisticRegression(max_iter=1000, penalty='l2')
model_l2.fit(X_train_scaled, y_train)
y_pred_l2 = model_l2.predict(X_test_scaled)
accuracy_l2 = accuracy_score(y_test, y_pred_l2)
print(f"L2 Regularized Logistic Regression Valixdation Accuracy: {accuracy_l2*100:.2f}%")

# list of models; print and iterate through their respective accuracy scores, confusion matrices, and classification reports
models = [model_l1, model_l2, svm.SVC(), RandomForestClassifier(), GradientBoostingClassifier()]

for model in models:
    # train the model using the training set
    model.fit(X_train, y_train)

    # use the trained model to predict the target variable in the validation set
    y_pred = model.predict(X_val)

    # calculate the accuracy of the model
    accuracy = accuracy_score(y_val, y_pred)

    # print the accuracy
    print(f"\nModel: {type(model).__name__}, Validation Accuracy: {accuracy*100:.2f}%")

    # confusion matrix and classification report
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("Classification Report:")
    print(classification_report(y_val, y_pred))





# generate a synthetic dataset; 1000 random features and labels
X, y = make_classification(n_samples=1000, random_state=42)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dummy classifier with a strategy of 'most_frequent'
# predicts the most frequent label in the training set
dummy_clf = DummyClassifier(strategy='most_frequent')

# train the dummy classifier
dummy_clf.fit(X_train, y_train)

# use the trained dummy classifier to predict the labels of the test data.
y_pred = dummy_clf.predict(X_test)

# calculate accuracy
# this is the proportion of correct predictions from the total number of predictions.
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {accuracy}")




# test for overfittting

# split data into training and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model on training data
model = RandomForestClassifier()  # for instance
model.fit(X_train, y_train)


# make predictions on training data
y_train_pred = model.predict(X_train)

# calculate accuracy on training data
train_acc = metrics.accuracy_score(y_train, y_train_pred)

# make predictions on test data
y_test_pred = model.predict(X_test)

# calculate accuracy on test data
test_acc = metrics.accuracy_score(y_test, y_test_pred)

print(f'Training Accuracy: {train_acc * 100}%')
print(f'Test Accuracy: {test_acc * 100}%')


# Create and fit the model on the training data (this is the 'model fit' step)
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set (this is the 'model prediction' step)
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))



# calculate the ROC-AUC score
roc_auc_val = roc_auc_score(y_val, y_pred_val)

print(f'Validation ROC-AUC score: {roc_auc_val:.2f}')

# graph the ROC score
fpr, tpr, _ = roc_curve(y_val, y_pred_val)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2  # Line width
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')  # This plots our baseline.
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic curve')
plt.legend(loc="lower right")
plt.show()



# # Drop columns where all values are NaN
# df = df.dropna(axis=1, how='all')

# # Now try creating the pairplot
# sns.pairplot(df)
# plt.show()



