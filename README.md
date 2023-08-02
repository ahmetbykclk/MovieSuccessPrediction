## Movie Success Prediction

This is a Python-based Movie Success Prediction project that uses Linear Regression to predict the IMDb scores of movies. The dataset is assumed to be in CSV format with columns containing various movie attributes, including the IMDb score, and movie titles.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [How it Works](#how-it-works)
- [Usage](#usage)

## Dataset

The dataset file dataset.csv contains various movie attributes, including the IMDb score and movie titles. The CSV format should have columns for these attributes.

You can download my dataset from this link:

https://www.kaggle.com/code/harshadeepvattikunta/predicting-movie-success/input

## Requirements

To run this project, you need the following dependencies:

- Python 3.x
- pandas
- scikit-learn
- matplotlib
  
You can install the required packages using the following command:

pip install pandas scikit-learn matplotlib

## How it Works

The Movie Success Prediction works as follows:

1- The dataset (CSV format) is loaded with various movie attributes, including the IMDb score and movie titles.

2- The total null values present in each column are displayed.

3- Samples with missing values are dropped from the dataset.

4- Certain columns that are not required or have multicollinearity are dropped.

5- Categorical columns are label-encoded to convert them into numerical features.

6- The features (X) and target (y) are prepared from the dataset, where the target is the IMDb score.

7- The data is split into training and testing sets using an 80-20 split.

8- The model, Linear Regression, is trained on the training set.

9- Predictions are made on the test set, and IMDb scores are classified into "Flop," "Average," or "Hit" categories.

10- The model is evaluated using Mean Squared Error and R-squared metrics.

11- The first 10 test predictions, actual values, and movie titles are printed.

12- Predicted IMDb scores and their classifications are visualized using a scatter plot.

## Usage

1- Clone the repository or download the moviessuccessprediction.py and dataset.csv files.

2- Make sure you have Python 3.x installed on your system.

3- Install the required dependencies by running pip install pandas scikit-learn matplotlib.

4- Run the moviessuccessprediction.py script.

The script will load the dataset, preprocess the data, train the Linear Regression model, and predict IMDb scores for movies. Additionally, it will display model evaluation metrics and visualize predicted IMDb scores.
