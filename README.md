# Sales Prediction using Python

This project demonstrates how to predict product sales based on advertising budgets across different media channels (TV, Radio, Newspaper) using machine learning in Python. 
The goal is to help businesses understand how advertising spend influences sales and to forecast future sales performance.

## Objective

The objective of this project is to predict product sales based on advertising budgets allocated to different media channels (TV, Radio, Newspaper) using machine learning.
Accurate sales prediction helps businesses optimize their advertising spend and make data-driven marketing decisions.

## Dataset

The dataset used is `Advertising.csv`, which contains the following columns:
- `TV`: Advertising budget spent on TV (in thousands of dollars)
- `Radio`: Advertising budget spent on Radio (in thousands of dollars)
- `Newspaper`: Advertising budget spent on Newspaper (in thousands of dollars)
- `Sales`: Units sold (in thousands)

> **Make sure your dataset is at `/content/Advertising.csv` or update the code with your file's path.**

## Project Steps

1. **Load and Explore the Data**  
   - Read the dataset and examine its structure and summary statistics.
   - Visualize the relationships between features and sales.

2. **Preprocess the Data**  
   - Check for missing values and clean the data if necessary.

3. **Feature Selection**  
   - Use TV, Radio, and Newspaper budgets as input features to predict sales.

4. **Train/Test Split**  
   - Split the data into training and testing sets (80% train, 20% test).

5. **Model Training**  
   - Train a Linear Regression model on the training data.

6. **Prediction and Evaluation**  
   - Predict sales on the test set.
   - Evaluate the model using Root Mean Squared Error (RMSE) and R² Score.

7. **Visualization**  
   - Visualize actual vs. predicted sales.

## Tools Used

- **Python**: Programming language used for implementation.
- **Pandas**: For data loading and manipulation.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **scikit-learn**: For machine learning (train/test split, Linear Regression, metrics).
- 
## How to Run

1. Ensure you have Python 3.x installed.
2. Install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place `Advertising.csv` in the `/content/` directory or update the file path in the code.
4. Run the script:
   ```bash
   python sales_prediction.py
   ```
   
## Example Output

```
Model Coefficients: [0.0458 0.1870 -0.0010]
Model Intercept: 2.9388893694594085
Root Mean Squared Error (RMSE): 1.95
R^2 Score: 0.90
```

A scatter plot comparing actual vs predicted sales will also be displayed.

## Outcome

- Built a Linear Regression model that predicts sales based on advertising budgets.
- Achieved a high R² score (close to 0.9 in typical runs), indicating that the model explains a significant portion of the variance in sales.
- Visualized the effectiveness of the model by comparing predicted and actual sales values.
- The project demonstrates how data-driven approaches can provide actionable insights for marketing and sales strategy in business.
- 


