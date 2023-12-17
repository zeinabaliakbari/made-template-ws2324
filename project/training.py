import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return model, accuracy, report

def plot_feature_importance(model, feature_names, title="Feature Importance"):
    # Get feature importances from the model
    if hasattr(model, 'coef_'):
        feature_importance = model.coef_[0]
    else:
        raise ValueError("Model doesn't have a 'coef_' attribute for feature importance.")

    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(title)
    plt.show()

def save_results_to_csv(results_df, filename='model_results.csv'):
    # Save the results to a CSV file
    results_df.to_csv(filename, index=False)

# Load preprocessed data (replace with actual file names)
diabetes_data = pd.read_csv('data/diabetes.csv')
kidney_data = pd.read_csv('data/KidneyDisease.csv')

# Define features and target for diabetes
diabetes_features = diabetes_data.drop('Outcome', axis=1)
diabetes_target = diabetes_data['Outcome']

# Define features and target for kidney disease
kidney_features = kidney_data.drop('Outcome', axis=1)
kidney_target = kidney_data['Outcome']

# Split the data into training and testing sets
diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(
    diabetes_features, diabetes_target, test_size=0.2, random_state=42
)

kidney_X_train, kidney_X_test, kidney_y_train, kidney_y_test = train_test_split(
    kidney_features, kidney_target, test_size=0.2, random_state=42
)

# Define logistic regression models
model_diabetes = LogisticRegression(max_iter=2000)
model_kidney = LogisticRegression(max_iter=1000)

# Train and evaluate models
model_diabetes, diabetes_accuracy, diabetes_report = train_and_evaluate_model(
    diabetes_X_train, diabetes_y_train, diabetes_X_test, diabetes_y_test, model_diabetes
)

model_kidney, kidney_accuracy, kidney_report = train_and_evaluate_model(
    kidney_X_train, kidney_y_train, kidney_X_test, kidney_y_test, model_kidney
)

# Save accuracy to a file
accuracy_df = pd.DataFrame({'Model': ['Diabetes', 'Kidney'], 'Accuracy': [diabetes_accuracy, kidney_accuracy]})
save_results_to_csv(accuracy_df, 'accuracy_results.csv')

# Save classification report to a file
report_df = pd.DataFrame({'Model': ['Diabetes', 'Kidney'], 'ClassificationReport': [diabetes_report, kidney_report]})
save_results_to_csv(report_df, 'classification_report_results.csv')

# Save feature importances to separate files
feature_importance_diabetes = pd.DataFrame({'Feature': diabetes_features.columns, 'Importance': model_diabetes.coef_[0]})
save_results_to_csv(feature_importance_diabetes, 'feature_importance_diabetes.csv')

feature_importance_kidney = pd.DataFrame({'Feature': kidney_features.columns, 'Importance': model_kidney.coef_[0]})
save_results_to_csv(feature_importance_kidney, 'feature_importance_kidney.csv')

# Create a visualization chart for model accuracy
fig, ax = plt.subplots()
models = ['Diabetes', 'Kidney']
accuracies = [diabetes_accuracy, kidney_accuracy]

ax.bar(models, accuracies, color=['blue', 'green'])
ax.set_ylabel('Accuracy')
ax.set_title('Model Accuracy Comparison')
plt.savefig('accuracy_chart.png')
plt.show()

# Visualize feature importance for each model
plot_feature_importance(model_diabetes, diabetes_features.columns, title="Diabetes Feature Importance")
plot_feature_importance(model_kidney, kidney_features.columns, title="Kidney Feature Importance")
