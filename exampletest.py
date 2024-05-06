import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# Importing the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data  = pd.read_csv('BankChurners.csv')


# Display the first few rows of the dataframe
print(data .head())
print(data.shape)
#drop columns
data = data.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
# Assuming df is your DataFrame
# Convert categorical columns using one-hot encoding
categorical_features = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
##from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Splitting the dataset
X = data.drop('Attrition_Flag', axis=1)  # Features
y = data['Attrition_Flag']  # Target variable, assuming 'Attrition_Flag' indicates churn

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Assuming 'Attrited Customer' means the customer has churned
y_train = (y_train == 'Attrited Customer').astype(int)
y_test = (y_test == 'Attrited Customer').astype(int)

# Now y_train and y_test contain 0s and 1s instead of string labels
dc= DecisionTreeClassifier(random_state=42)
dc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dc.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
from sklearn.metrics import roc_curve, roc_auc_score
# Predict probabilities for the test set
y_probs = dc.predict_proba(X_test)[:, 1]  # Probability scores for the positive class
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, roc_auc_score
# Model evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred_binary))
# Calculate FPR, TPR, and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Calculate the AUC
roc_auc = auc(fpr, tpr)
print(f"ROC AUC: {roc_auc}")
import matplotlib.pyplot as plt

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
import matplotlib.pyplot as plt

# Assuming 'dc' is your trained Decision Tree model
feature_importances = dc.feature_importances_
features = X_train.columns

# Create a DataFrame to view features and their importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.show()
# Assuming 'y_probs' contains the probabilities of churning for the test set
# Let's create a DataFrame for easy manipulation
test_results = pd.DataFrame({
    'Churn_Probability': y_probs,
    'Actual': y_test
})

# Define risk categories based on churn probability
def risk_category(prob):
    if prob > 0.66:
        return 'High Risk'
    elif prob > 0.33:
        return 'Medium Risk'
    else:
        return 'Low Risk'

# Apply function to categorize risks
test_results['Risk_Category'] = test_results['Churn_Probability'].apply(risk_category)

# Save this data to a CSV file
high_risk_customers.to_csv('high_risk_customers.csv', index=False)

print("High risk customers saved to 'high_risk_customers.csv'.")
# Count the number of customers in each category, ensuring all categories are represented
risk_counts = test_results['Risk_Category'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk'], fill_value=0)

# Labels, sizes, colors, and explosion settings
labels = risk_counts.index
sizes = risk_counts.values
colors = ['#99ff99', '#ffcc99', '#ff6666']  # Green for low, orange for medium, red for high
explode = (0, 0, 0.1)  # only explode the slice for 'High Risk'

# Plotting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Churn Risk Distribution in Test Set')
plt.show()
import pandas as pd

# Assuming you have a DataFrame 'test_results' from previous steps which includes 'Churn_Probability'
# Recreate the 'Risk_Category' in case it's not already there
def risk_category(prob):
    if prob > 0.66:
        return 'High Risk'
    elif prob > 0.33:
        return 'Medium Risk'
    else:
         return 'Low Risk'
test_results = X_test.copy()
test_results['Churn_Probability'] = y_probs
test_results['Risk_Category'] = test_results['Churn_Probability'].apply(risk_category)

# Filter to get only High Risk customers
high_risk_customers = test_results[test_results['Risk_Category'] == 'High Risk']

# Save this data to a CSV file, ensuring to include 'CLIENTNUM'
high_risk_customers.to_csv('ID_high_risk_customers.csv', index=False)

print("ID High risk customers saved to 'ID _ high_risk_customers.csv'.")