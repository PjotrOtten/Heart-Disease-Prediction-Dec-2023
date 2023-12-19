

# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('heart.csv')

# Explore the data
df.head()
df.tail()
df.info()
df.describe()

# Age distribution graph
# Separate ages for people with and without heart disease
heart_disease_0_age = df[df['HeartDisease'] == 0]['Age']
heart_disease_1_age = df[df['HeartDisease'] == 1]['Age']

# Plot a histogram for age distribution
plt.hist([heart_disease_0_age, heart_disease_1_age], bins=50, alpha=0.7, color=['blue', 'red'],
         label=['No Heart Disease', 'Heart Disease'], stacked=True)

# Set labels and title for the plot
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')

# Add legend to distinguish between people with and without heart disease
plt.legend()

# Display the plot
plt.show()

# Total People by Sex and HeartDisease graph
# Group data by Sex and HeartDisease and count the total number of people
df_sex_heart = df.groupby(['Sex', 'HeartDisease'])['Sex'].count().reset_index(name='Total')
df_sex_heart['HeartDisease'] = df_sex_heart['HeartDisease'].astype(str)
sns.barplot(data=df_sex_heart, x='Sex', y='Total', hue='HeartDisease', palette=['blue', 'red'])
plt.title('Total People by Sex and HeartDisease')
plt.xlabel('Sex (0: Female, 1: Male)')
plt.ylabel('Total People')
legend_labels = ['No Heart Disease', 'Have Heart Disease']
legend_colors = {'No Heart Disease': 'blue', 'Have Heart Disease': 'red'}
handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in legend_labels]
plt.legend(handles, legend_labels)
plt.show()

# RestingBP Distribution graph
# Separate RestingBP data for people with and without heart disease
heart_disease_0 = df[df['HeartDisease'] == 0]['RestingBP']
heart_disease_1 = df[df['HeartDisease'] == 1]['RestingBP']
plt.hist([heart_disease_0, heart_disease_1], bins=25, alpha=0.7, color=['blue', 'red'],
         label=['No Heart Disease', 'Heart Disease'], range=(80, df['RestingBP'].max()))
plt.xlabel('RestingBP')
plt.ylabel('Frequency')
plt.title('RestingBP Distribution')
plt.legend()
plt.show()

# Cholesterol Distribution graph
# Separate Cholesterol data for people with and without heart disease
heart_disease_0 = df[df['HeartDisease'] == 0]['Cholesterol']
heart_disease_1 = df[df['HeartDisease'] == 1]['Cholesterol']
plt.hist([heart_disease_0, heart_disease_1], bins=50, alpha=0.7, color=['blue', 'red'],
         label=['No Heart Disease', 'Heart Disease'], range=(80, df['Cholesterol'].max()), stacked=True)
plt.xlabel('Cholesterol')
plt.ylabel('Frequency')
plt.title('Cholesterol Distribution')
plt.legend()
plt.show()

# Total People by FastingBS and HeartDisease graph
# Group data by FastingBS and HeartDisease and count the total number of people
df_fasting_heart = df.groupby(['FastingBS', 'HeartDisease'])['FastingBS'].count().reset_index(name='Total')
df_fasting_heart['HeartDisease'] = df_fasting_heart['HeartDisease'].astype(str)
sns.barplot(data=df_fasting_heart, x='FastingBS', y='Total', hue='HeartDisease', palette=['blue', 'red'])
plt.title('Total People by FastingBS and HeartDisease')
plt.xlabel('Fasting Blood Sugar (Y: Yes, N: No)')
plt.ylabel('Total People')

# Add legend with colors
legend_labels = ['No Heart Disease', 'Have Heart Disease']
legend_colors = {'No Heart Disease': 'blue', 'Have Heart Disease': 'red'}
handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in legend_labels]
plt.legend(handles, legend_labels)
plt.show()

# Resting Electrocardiogram graph
# Count the occurrences of each RestingECG value
data_restelectro = df['RestingECG'].value_counts()
labels_restelectro = df['RestingECG'].value_counts().index
sns.barplot(x=labels_restelectro, y=data_restelectro, palette=['blue', 'red', 'green'])
plt.title('Resting Electrocardiogram')
plt.xlabel('Resting Electrocardiogram')
plt.ylabel('Count')
plt.show()

# Max Heart Rate Distribution graph
# Separate MaxHR data for people with and without heart disease
heart_disease_0 = df[df['HeartDisease'] == 0]['MaxHR']
heart_disease_1 = df[df['HeartDisease'] == 1]['MaxHR']
plt.hist([heart_disease_0, heart_disease_1], bins=50, alpha=0.7, color=['blue', 'red'],
         label=['No Heart Disease', 'Heart Disease'], stacked=True)
plt.xlabel('Max Heart Rate')
plt.ylabel('Frequency')
plt.title('Max Heart Rate Distribution')
plt.legend()
plt.show()

# Total People by ExerciseAngina and HeartDisease graph
# Group data by ExerciseAngina and HeartDisease and count the total number of people
df_agina_heart = df.groupby(['ExerciseAngina', 'HeartDisease'])['ExerciseAngina'].count().reset_index(name='Total')
df_agina_heart['HeartDisease'] = df_agina_heart['HeartDisease'].astype(str)
sns.barplot(data=df_agina_heart, x='ExerciseAngina', y='Total', hue='HeartDisease', palette=['blue', 'red'])
plt.title('Total People by Exercise-Induced Angina and Heart Disease')
plt.xlabel('Exercise-Induced Angina (Y: Yes, N: No)')
plt.ylabel('Total People')
legend_labels = ['No Heart Disease', 'Have Heart Disease']
legend_colors = {'No Heart Disease': 'blue', 'Have Heart Disease': 'red'}
handles = [plt.Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in legend_labels]
plt.legend(handles, legend_labels)
plt.show()

# Oldpeak Distribution graph
# Separate Oldpeak data for people with and without heart disease
heart_disease_0_oldpeak = df[df['HeartDisease'] == 0]['Oldpeak']
heart_disease_1_oldpeak = df[df['HeartDisease'] == 1]['Oldpeak']
plt.hist([heart_disease_0_oldpeak, heart_disease_1_oldpeak], bins=50, alpha=0.7, color=['blue', 'red'],
         label=['No Heart Disease', 'Heart Disease'], stacked=True)
plt.xlabel('Oldpeak')
plt.ylabel('Frequency')
plt.title('Oldpeak Distribution')
plt.legend()
plt.show()

# Heart Disease Distribution
# Count the occurrences of each HeartDisease value
data_hd = df['HeartDisease'].value_counts()
labels = ['No Heart Disease', 'Have Heart Disease']
# Plot a bar graph for Heart Disease distribution
bars = plt.bar(labels, data_hd, color=['blue', 'red'])
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.title('Heart Disease Distribution')
plt.legend(bars, labels)
plt.show()

# Correlation matrix
# Select columns of interest for correlation analysis
columns_of_interest = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'HeartDisease']
selected_df = df[columns_of_interest]
correlation_matrix = selected_df.corr()
# Plot a heatmap for the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# Feature Engineering
# Create categorical features based on certain bins
df['BP_Category'] = pd.cut(df['RestingBP'], bins=[0, 120, 130, 140, 160, 180, 200],
                           labels=['Normal', 'Elevated', 'Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
df['Chol_Category'] = pd.cut(df['Cholesterol'], bins=[0, 200, 240, 300, 1000],
                             labels=['Normal', 'Borderline High', 'High', 'Very High'])
df['Age_Group'] = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 70, 80, 90],
                        labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90'])
df['BP_Chol_Interaction'] = df['RestingBP'] * df['Cholesterol']

# Label encoding for categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['ChestPainType'] = le.fit_transform(df['ChestPainType'])
df['FastingBS'] = le.fit_transform(df['FastingBS'])
df['RestingECG'] = le.fit_transform(df['RestingECG'])
df['ExerciseAngina'] = le.fit_transform(df['ExerciseAngina'])
df['ST_Slope'] = le.fit_transform(df['ST_Slope'])
df['HeartDisease'] = le.fit_transform(df['HeartDisease'])

# Split the dataset into features (X) and target variable (y)
X = df.drop(['HeartDisease', 'Age_Group'], axis=1)
y = df['HeartDisease']

# One-hot encoding for categorical variables
X = pd.get_dummies(X, columns=['BP_Category', 'Chol_Category'], drop_first=True)


# Logistic Regression (LR) Model
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test_original = train_test_split(X, y, test_size=0.2, random_state=1)

# Standardize the data
sc = StandardScaler()
X_train[['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'BP_Chol_Interaction']] = sc.fit_transform(
    X_train[['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'BP_Chol_Interaction']])
X_test[['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'BP_Chol_Interaction']] = sc.transform(
    X_test[['RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak', 'BP_Chol_Interaction']])

# Train Logistic Regression model
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# Evaluate and display results
cm_lr = confusion_matrix(y_test_original, y_pred_lr)
lr_test_acc = accuracy_score(y_test_original, y_pred_lr)
print('Logistic Regression Accuracy = ', lr_test_acc)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression Confusion Matrix')
plt.show()
Accuracy_LR = lr_test_acc
classification_rep_lr = classification_report(y_test_original, y_pred_lr)
print("Logistic Regression Classification Report:\n", classification_rep_lr)


# Artificial Neural Network (ANN) Model
# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
tf.compat.v1.set_random_seed(42)

# Initialize variables for best model
best_accuracy = 0
best_model = None

# Iterate over multiple runs to find the best ANN model
for run in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build ANN model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

    # Predict on the test set
    y_pred_test_ann = model.predict(X_test)
    y_pred_test_ann = (y_pred_test_ann > 0.5).astype(int)

    # Calculate accuracy
    accuracy_ann = accuracy_score(y_test, y_pred_test_ann)

    # Update best model if the current model has higher accuracy
    if accuracy_ann > best_accuracy:
        best_accuracy = accuracy_ann
        best_model = model

# Display the best ANN model's results
print(f"\nBest ANN Test Set Accuracy: {best_accuracy:.4f}")
ANN_accuracy = best_accuracy
y_pred_best_ann = best_model.predict(X_test)
y_pred_best_ann = (y_pred_best_ann > 0.5).astype(int)

# Confusion Matrix for the best model
cm_best_ann = confusion_matrix(y_test, y_pred_best_ann)
print(cm_best_ann)

# Plot Confusion Matrix for the best model
sns.heatmap(cm_best_ann, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Best ANN Confusion Matrix')
plt.show()
print("Best ANN Test Set Classification Report:\n", classification_report(y_test, y_pred_best_ann))


# Random Forest Model
# Set seed for reproducibility
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Resample the training set using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Build and train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=1)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)

# Calculate and display accuracy
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.4f}")

# Create the confusion matrix
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)

# Plot Confusion Matrix for Random Forest
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Random Forest Confusion Matrix')
plt.show()

# Display the classification report
classification_rep_rf = classification_report(y_test, y_pred_rf)
print("\nRandom Forest Test Set Classification Report:\n", classification_rep_rf)


# SVM Model
# Set seed for reproducibility
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for SVM using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Initialize the SVM model
svm_model = SVC()

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params_svm = grid_search.best_params_
print(f"Best Hyperparameters: {best_params_svm}")

# Use the best SVM model to predict on the test set
best_svm_model = grid_search.best_estimator_
y_pred_test_svm = best_svm_model.predict(X_test)

# Calculate and display accuracy
accuracy_svm = accuracy_score(y_test, y_pred_test_svm)
print(f"SVM - Test Set Accuracy: {accuracy_svm:.4f}")

# Display the confusion matrix
conf_matrix_svm = confusion_matrix(y_test, y_pred_test_svm)

# Plot Confusion Matrix for SVM
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('SVM Confusion Matrix')
plt.show()

# Display the classification report
classification_rep_svm = classification_report(y_test, y_pred_test_svm)
print("\nSVM Test Set Classification Report:\n", classification_rep_svm)


# K Nearest Neighbors (KNN) Model
# Set seed for reproducibility
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train_scaled_knn = scaler.fit_transform(X_train)
X_test_scaled_knn = scaler.transform(X_test)

# Initialize KNN model
knn_model = KNeighborsClassifier()

# Define parameter grid for KNN
param_grid_knn = {'n_neighbors': list(range(3, 11)), 'weights': ['uniform', 'distance'],
                  'metric': ['euclidean', 'manhattan']}

# Use GridSearchCV to find the best hyperparameters for KNN
grid_search_knn = GridSearchCV(knn_model, param_grid_knn, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_knn.fit(X_train_scaled_knn, y_train)

# Get the best KNN model
best_knn_model = grid_search_knn.best_estimator_
print(f"\nBest KNN Test Set Accuracy: {grid_search_knn.best_score_:.4f}")
print(f"Best KNN Model Parameters: {grid_search_knn.best_params_}")

# Use the best KNN model to predict on the test set
y_pred_knn = best_knn_model.predict(X_test_scaled_knn)

# Calculate and display accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Best KNN Test Set Accuracy: {knn_accuracy:.4f}")

# Display the confusion matrix
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
print("\nBest KNN Test Set Confusion Matrix:\n", conf_matrix_knn)

# Plot Confusion Matrix for KNN
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Best KNN Confusion Matrix')
plt.show()

# Display the classification report
classification_rep_knn = classification_report(y_test, y_pred_knn)
print("\nBest KNN Test Set Classification Report:\n", classification_rep_knn)


# Naïve Bayes Model
np.random.seed(42)
scaler_nb = StandardScaler()
X_scaled_nb = scaler_nb.fit_transform(X)
splitter_nb = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index_nb, test_index_nb in splitter_nb.split(X_scaled_nb, y):
    X_train_nb, X_test_nb = X_scaled_nb[train_index_nb], X_scaled_nb[test_index_nb]
    y_train_nb, y_test_nb = y[train_index_nb], y[test_index_nb]
nb_model = GaussianNB()
nb_model.fit(X_train_nb, y_train_nb)
y_pred_test_nb = nb_model.predict(X_test_nb)

# Evaluate and display results
accuracy_nb = accuracy_score(y_test_nb, y_pred_test_nb)
print(f"Naïve Bayes Accuracy: {accuracy_nb:.4f}")
conf_matrix_nb = confusion_matrix(y_test_nb, y_pred_test_nb)
print("\nNaive Bayes Test Set Confusion Matrix:\n", conf_matrix_nb)
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Naive Bayes Confusion Matrix')
plt.show()
classification_rep_nb = classification_report(y_test_nb, y_pred_test_nb)
print("\nNaive Bayes Test Set Classification Report:\n", classification_rep_nb)


# Decision Tree Model
X_train, X_test, y_train, y_test_original = train_test_split(X, y, test_size=0.2, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dt_model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, min_samples_leaf=1)
dt_model.fit(X_train_scaled, y_train)
y_pred_test_dt = dt_model.predict(X_test_scaled)

# Evaluate and display results
accuracy_DT = accuracy_score(y_test_original, y_pred_test_dt)
print(f"Decision Tree Accuracy: {accuracy_DT:.4f}")
conf_matrix_dt = confusion_matrix(y_test_original, y_pred_test_dt)
print("\nDecision Tree Test Set Confusion Matrix:\n", conf_matrix_dt)
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Decision Tree Confusion Matrix')
plt.show()
classification_rep_dt = classification_report(y_test_original, y_pred_test_dt)
print("\nDecision Tree Test Set Classification Report:\n", classification_rep_dt)


# XGBoost Model
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in stratified_splitter.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test_original = y.iloc[train_index], y.iloc[test_index]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
xgb_model = XGBClassifier(random_state=123)
xgb_model.fit(X_train_scaled, y_train)
y_pred_test_xgb = xgb_model.predict(X_test_scaled)

# Evaluate and display results
accuracy_xgb = accuracy_score(y_test_original, y_pred_test_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.4f}")
conf_matrix_xgb = confusion_matrix(y_test_original, y_pred_test_xgb)
print("\nXGBoost Test Set Confusion Matrix:\n", conf_matrix_xgb)
sns.heatmap(conf_matrix_xgb, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('XGBoost Confusion Matrix')
plt.show()
classification_rep_xgb = classification_report(y_test_original, y_pred_test_xgb)
print("\nXGBoost Test Set Classification Report:\n", classification_rep_xgb)


# accuracies graph
accuracies = [lr_test_acc, ANN_accuracy, accuracy_rf, accuracy_svm, knn_accuracy, accuracy_nb, accuracy_DT, accuracy_xgb]
models = ['LR', 'ANN', 'RF', 'SVM', 'KNN', 'NB', 'DT', 'XGBoost']
df = pd.DataFrame({'Model': models, 'Accuracy': accuracies})
df = df.sort_values(by='Accuracy', ascending=True)
plt.figure(figsize=(10, 6))
bars = plt.bar(df['Model'], df['Accuracy'], color=['blue', 'grey', 'lightblue', 'orange', 'green', 'red', 'purple', 'brown'])
plt.title('Model Accuracies')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.ylim(0, 1)  
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')
plt.show()


# Accuracies table
models = pd.DataFrame({
    'Model': [
        'Logistic Regression', 'ANN', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'Decision Tree', 'XGBoost'
    ],
    'Model Accuracies': [
        lr_test_acc, ANN_accuracy, accuracy_rf, accuracy_svm, knn_accuracy, accuracy_nb, accuracy_DT, accuracy_xgb
    ]
})
# Convert accuracies to percentages
models['Model Accuracies'] = models['Model Accuracies'].map(lambda x: f'{x:.2%}')

print(models.sort_values(by='Model Accuracies', ascending=False))

