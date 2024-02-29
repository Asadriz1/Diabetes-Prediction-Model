from preprocessing import load_data, explore_data, split_features_and_target, split_data, standardize_data
from model import train_svm, evaluate_model
from prediction import make_prediction

# Change this path to your local path where the dataset is stored
dataset_path = '/Users/asadrizvi/Desktop/diabetes-prediction/Dataset/diabetes.csv'

# Load and explore the data
diabetes_data = load_data(dataset_path)
explore_data(diabetes_data)

# Preprocess the data
X, Y = split_features_and_target(diabetes_data)
standardized_X, scaler = standardize_data(X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = split_data(standardized_X, Y)

# Train the model
svm_classifier = train_svm(X_train, Y_train)

# Evaluate the model
training_accuracy = evaluate_model(svm_classifier, X_train, Y_train)
print(f'Accuracy score of the training data: {training_accuracy}')

test_accuracy = evaluate_model(svm_classifier, X_test, Y_test)
print(f"Accuracy score of the test data: {test_accuracy}")


