import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from google.colab import files
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Upload the dataset file
uploaded = files.upload()

# Load your dataset and prepare X and y
# Assuming df is your pandas DataFrame with the data
file_name = next(iter(uploaded))
df = pd.read_excel(file_name, sheet_name='data', engine='openpyxl')

# Create a binary target variable 'Dropped Out'
df['Dropped Out'] = df['Level of Education'].apply(lambda x: 1 if x == 'High School' else 0)

# Prepare X and y
X = df[['Last Name','Cultural Identity', 'Gender','Age', 'Government Funding', 'Type of Educational Institute','Community Involvement','Level of Education','Language Proficiency','Dropped Out']]
y = df['Dropped Out']

# One-hot encode categorical features
categorical_features = ['Last Name','Cultural Identity', 'Gender','Age', 'Government Funding', 'Type of Educational Institute','Community Involvement','Level of Education','Language Proficiency','Dropped Out']
categorical_transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = categorical_transformer.fit_transform(X)

# Split the data into training and testing sets with stratified split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

# Create an instance of the logistic regression model with class_weight='balanced'
model = LogisticRegression(class_weight='balanced')

# Choose the desired number of features to keep based on your research objectives
desired_num_features = 5

# Implement RFE algorithm using cross_val_score to calculate F1-score during feature selection
rfe = RFE(model, n_features_to_select=desired_num_features)
f1_scores = cross_val_score(rfe, X_train, y_train, cv=5, scoring='f1')

# Get the average F1-score across the cross-validation folds
average_f1 = f1_scores.mean()

print("Average F1-score during feature selection:", average_f1)

# Fit the model on the selected features
rfe.fit(X_train, y_train)
X_train_selected = X_train[:, rfe.support_]
model.fit(X_train_selected, y_train)

# Predict the target variable (Dropped Out) on the testing set
X_test_selected = X_test[:, rfe.support_]
y_pred = model.predict(X_test_selected)

# Calculate the F1-score for the 'Dropped Out' class
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("F1-score for 'Dropped Out' instances:",f1)
print("Accuracy:", accuracy)