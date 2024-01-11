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


### Get feature importances from the model _accuracy, _accuracy
def kidney_feature_importance(model, feature_names, title):
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

def diabetes_feature_importance(model, feature_names,title):
    # Accessing the coefficients (weights) of the first layer
    first_layer_weights = model.coefs_[0]

    # Taking the absolute values to emphasize the importance, assuming positive and negative values are both important
    importance = abs(first_layer_weights).sum(axis=1)

    # Creating a DataFrame for better visualization
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

    # Sorting features by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(title)
    plt.show()


### Save the results to a CSV file
def save_results_to_csv(results_df, filename='model_results.csv'):
    
    results_df.to_csv(filename, index=False)


def train_neural_network(X, y, hidden_layer_sizes=(8, 4), learning_rate_init=0.001, max_iter=1000, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the MLP model
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, learning_rate_init=learning_rate_init, max_iter=max_iter, random_state=random_state)

    # Train the model
    mlp_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = mlp_model.predict(X_test)

    # Evaluate the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    #report = classification_report(y_test, y_pred)

    # Return the trained model
    return mlp_model, accuracy

 



# Load your dataset
def plot_accuracy_comparison(accuracies, labels):
    # Plotting the bar chart
    plt.bar(labels, accuracies, color=['blue', 'orange'])
    plt.ylim(0, 1)  # Setting y-axis limits to represent accuracy percentage
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.show()    




#################################################  Kidney model  #####################################################
 
kidney_data = pd.read_csv('KidneyDisease.csv')     
# Define features and target for kidney disease
kidney_features = kidney_data.drop('Outcome', axis=1)
kidney_target = kidney_data['Outcome']

# Split the data into training and testing sets

kidney_X_train, kidney_X_test, kidney_y_train, kidney_y_test = train_test_split(
    kidney_features, kidney_target, test_size=0.2, random_state=42
)

# Define logistic regression models
 
model_kidney = LogisticRegression(max_iter=2000)

model_kidney, kidney_accuracy, kidney_report = train_and_evaluate_model(
    kidney_X_train, kidney_y_train, kidney_X_test, kidney_y_test, model_kidney
)




#################################################  diabetes model    #####################################################
diabetes_data = pd.read_csv('diabetes.csv')
# Assume 'target' is the name of your target variable
X2 = diabetes_data.drop('Outcome', axis=1)
y2 = diabetes_data['Outcome']
#print('The information about modeling diabetes')
diabetes_features = X2.columns.tolist()
# Train the neural network 
diabetes_model, diabetes_accuracy = train_neural_network(X2, y2)

###################################################   Results  ##########################################################


print('********************************************')
print(f'diabetes_accuracy: {diabetes_accuracy}')
 

 
print(f'kidney_accuracy: {kidney_accuracy}')
print('********************************************')
 
 


plot_accuracy_comparison([kidney_accuracy, diabetes_accuracy], ['kidney_accuracy', 'diabetes_accuracy'])


 
kidney_feature_importance(model_kidney, kidney_features.columns, title="Kidney Feature Importance")



diabetes_feature_importance(diabetes_model, diabetes_features, "diabetes Feature Importance")
