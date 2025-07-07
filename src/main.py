
from features.load_data import load_iris_data
from features.preprocess_data import split_data, standardize_data
from models.train_model import train_logistic_regression
from models.evaluate_model import evaluate_model
from visualization.visualize_data import visualize_data

def main():
    # Load the Iris dataset
    X, y = load_iris_data()

    # Explore the dataset
    print("First 5 rows of the dataset:")
    print(X.head())

    print("\nSummary statistics of the dataset:")
    print(X.describe())

    # Visualize the dataset
    visualize_data(X, y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Standardize the features
    X_train_scaled, X_test_scaled = standardize_data(X_train, X_test)

    # Train a Logistic Regression model
    model = train_logistic_regression(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(model, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
