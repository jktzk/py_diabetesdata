# BMI as a Predictor for Diabetes
## Overview
This project explores BMI as a predictor for diabetes using a dataset from Kaggle. The dataset
contains variables such as age, hypertension, HbA1c levels, blood glucose, and diabetes status.
Custom Python code was written to preprocess the data, conduct regression analysis, and visualize
relationships, with a focus on understanding the correlation between BMI and diabetes.
## Features Implemented
- Data preprocessing:
 - Handling missing or improperly formatted data.
 - Separating variables into categorical and numerical types.
- Statistical calculations:
 - Means, standard deviations, and maximum values.
 - Custom implementation of regression models.
- Data visualization:
 - Generation of a regression line for BMI and diabetes using Matplotlib.
- Summary statistics:
 - Displaying formatted tables with key insights.
## Technologies Used
- **Python**: Core language for development.
- **Libraries**:
 - `numpy`: Sparse usage for numerical operations.
 - `matplotlib`: Visualization of regression results.
 - `csv`: Reading and processing the dataset.
 - `math` and `copy`: Utilities for custom calculations and data handling.
## How to Set Up / Get Started
1. **Dataset**: Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset). Save it as
`diabetes.csv` in the project directory.
2. **Dependencies**: Install required libraries using:
 ```bash
 pip install numpy matplotlib
 ```
3. **Run the Code**: Execute the script using:
 ```bash
 python diabetes_study.py
 ```

## Usage
The script preprocesses the data, performs regression analysis, and visualizes the relationship
between BMI and diabetes. Results include a regression plot and summary statistics such as
means, standard deviations, and category counts.
