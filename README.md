# Fraud Credit Card Detection

This project utilizes machine learning techniques to detect fraudulent credit card transactions. By analyzing transaction data, the model aims to accurately identify fraudulent activities, providing a robust solution for fraud detection in financial systems.

## Dataset

To run this project, you'll need to download the dataset from the following link:

[Download Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud/download)

**Note:** Place the downloaded CSV file in the same directory as your main and test files.

## Project Overview

The project consists of two main files:

### `main.py`

This file contains the main code for training the machine learning model and evaluating its performance. It performs the following tasks:

- Loads the dataset (`creditcard.csv`) into a Pandas DataFrame.
- Explores and visualizes the dataset to understand its structure and characteristics.
- Prepares the data for training by splitting it into features (X) and labels (Y).
- Divides the dataset into training and testing sets using Scikit-learn's `train_test_split` function.
- Builds a Random Forest Classifier model and trains it on the training data.
- Evaluates the model's performance using various metrics such as accuracy, precision, recall, F1-score, and Matthews correlation coefficient.
- Generates a confusion matrix to visualize the classification results.

### `test.py`

This file contains code for testing the trained model on a separate dataset or real-time transactions. It performs similar tasks to `main.py` but is focused on applying the model to new data rather than training it. 

## Usage

1. **Download Dataset:** Download the dataset from the provided link and place it in the project directory.
2. **Install Dependencies:** Make sure you have all the required Python libraries installed, including NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.
3. **Run `main.py`:** Execute the `main.py` file to train the machine learning model and evaluate its performance.
4. **Run `test.py`:** Optionally, run the `test.py` file to test the trained model on new data or real-time transactions.

## Creator

- [Jeet Chauhan](https://github.com/JeetChauhan17)

## License

This project is licensed under the [MIT License](LICENSE).
