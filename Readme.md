# Coco Pufs Project

![Coco Pufs](https://example.com/coco_pufs_image.png)

## Table of Contents

1. [Introduction](#introduction)
2. [Project Overview](#project-overview)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributions](#contributions)
9. [License](#license)
10. [Contact](#contact)

---

## Introduction

Welcome to the Coco Pufs Project! This project aims to predict customer preferences for the Coco Pufs cereal using a Linear Classifier Logistic Regression model. By analyzing various features, the model can forecast whether a customer will prefer Coco Pufs over other cereals.

## Project Overview

The primary objective of this project is to develop a predictive model using Logistic Regression to classify customer preferences based on their demographic and behavioral data. This model helps understand which features are most significant in predicting customer choices and can aid in targeted marketing strategies.

## Dataset

The dataset used for this project contains customer data, including demographic information, purchase history, and preferences. The key features in the dataset are:

- **CustomerID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Gender of the customer (Male/Female).
- **AnnualIncome**: Annual income of the customer.
- **PurchaseFrequency**: How often the customer buys cereals.
- **Preference**: Binary variable indicating if the customer prefers Coco Pufs (1) or not (0).

### Sample Data

| CustomerID | Age | Gender | AnnualIncome | PurchaseFrequency | Preference |
|------------|-----|--------|--------------|-------------------|------------|
| 1          | 25  | Male   | 50000        | 10                | 1          |
| 2          | 30  | Female | 60000        | 15                | 0          |
| 3          | 22  | Male   | 45000        | 5                 | 1          |

## Model

The model used in this project is a Logistic Regression classifier, which is suitable for binary classification tasks. Logistic Regression predicts the probability that a given input point belongs to a certain class (e.g., preference for Coco Pufs).

### Model Pipeline

1. **Data Preprocessing**: Handle missing values, encode categorical variables, and scale numerical features.
2. **Feature Selection**: Select the most relevant features for prediction.
3. **Model Training**: Train the Logistic Regression model on the training dataset.
4. **Model Evaluation**: Evaluate the model using accuracy, precision, recall, and F1 score.
5. **Prediction**: Use the trained model to predict customer preferences.

### Model Parameters

- **Regularization**: L2 (Ridge) regularization to avoid overfitting.
- **Solver**: ‘liblinear’ for small datasets and binary classification.
- **Threshold**: Decision threshold set to 0.5 for binary classification.

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/coco-pufs-project.git
    cd coco-pufs-project
    ```

2. **Create and activate a virtual environment** (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Prepare the dataset**: Ensure that your dataset is in the `data` directory.

2. **Run the Jupyter Notebook**:

    ```bash
    jupyter notebook
    ```

    Open `Coco_Pufs_Prediction.ipynb` and run the cells to preprocess the data, train the model, and make predictions.

3. **Make Predictions**:

    Use the provided script to make predictions on new customer data:

    ```bash
    python predict.py --input data/new_customers.csv --output results/predictions.csv
    ```

    **Parameters**:
    - `--input`: Path to the new customer data file.
    - `--output`: Path to save the prediction results.

## Results

### Model Performance

The Logistic Regression model achieved the following performance metrics on the test dataset:

- **Accuracy**: 85%
- **Precision**: 80%
- **Recall**: 78%
- **F1 Score**: 79%

### Confusion Matrix

![Confusion Matrix](https://example.com/confusion_matrix.png)

The model shows good performance in predicting customer preferences, with a balance between precision and recall.

### Feature Importance

The most significant features influencing the predictions were:

- **Annual Income**: Higher income indicates a higher likelihood of preferring Coco Pufs.
- **Purchase Frequency**: Customers who purchase cereals more frequently are more likely to prefer Coco Pufs.
- **Age**: Younger customers showed a higher preference for Coco Pufs.

## Contributions

Contributions are welcome! To contribute to the project, follow these steps:

1. **Fork the repository**.
2. **Create a new branch** for your feature or bug fix.
3. ** Please implement your changes and commit them**.
4. **Push your changes to your fork**.
5. **Submit a pull request** to the main repository.


## Contact

For questions or suggestions, please contact:

- **Name**: Dhruv Bansal
- **Email**: dhrubb22@iitk.ac.in
- **GitHub**: [@janedoe](https://github.com/janedoe)

---
