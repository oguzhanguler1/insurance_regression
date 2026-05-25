# Medical Insurance Cost Prediction

## Project Overview

This project aims to predict medical insurance charges using personal and health-related information such as age, sex, BMI, number of children, smoking status, and region.

The main goal of this project is to compare different regression models and evaluate which model predicts medical insurance costs more accurately.

Three regression models were trained and tested:

- Multiple Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)

The models were evaluated using MAE, MSE, RMSE, and R² Score.

The results showed that Random Forest Regressor and Support Vector Regression performed better than Multiple Linear Regression. Although SVR achieved the lowest MAE, Random Forest achieved a slightly better RMSE and R² Score. Since Random Forest also provides feature importance and is easier to interpret, it was selected as the final model.

---

## Dataset

The dataset contains medical insurance information for different individuals.

Each row represents one person and includes demographic and health-related features.

### Features

| Feature | Description |
|---|---|
| age | Age of the person |
| sex | Gender of the person |
| bmi | Body Mass Index |
| children | Number of children covered by insurance |
| smoker | Smoking status of the person |
| region | Residential region |
| charges | Medical insurance cost |

### Target Variable

The target variable is:

```text
charges
```

The `charges` column represents the medical insurance cost that the models try to predict.

---

## Technologies Used

This project was developed using the following Python libraries:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn

The project was implemented in a Jupyter Notebook.

---

## Project Workflow

The project follows these steps:

1. Load the dataset
2. Inspect the dataset
3. Check for missing values
4. Perform exploratory data analysis
5. Analyze the relationship between features and insurance charges
6. Check for possible outliers
7. Encode categorical variables
8. Split the dataset into training and test sets
9. Train regression models
10. Evaluate model performance
11. Compare model results
12. Visualize actual vs predicted values
13. Select the final model

---

## Data Inspection

The dataset was first loaded into a pandas DataFrame.

Basic inspection commands were used:

```python

df.info()
df.describe()
df.isnull().sum()
```

These commands helped check:

- Column names
- Data types
- Missing values
- Basic statistical information
- General structure of the dataset

No missing values were found in the dataset.

---

## Exploratory Data Analysis

Exploratory data analysis was performed to better understand the dataset and the relationship between the input features and the target variable.

Some important observations:

- Smoking status has a strong effect on medical insurance charges.
- Smokers generally have much higher insurance charges than non-smokers.
- Age has a positive relationship with insurance charges.
- BMI also affects insurance charges.
- Region has a weaker relationship with insurance charges compared to smoking status.
- The `charges` column contains high-cost values, but these values were not removed because they may represent valid high-risk insurance cases.

Example analysis:

```python
df.groupby("smoker")["charges"].mean()
df.groupby("region")["charges"].mean()
df.corr(numeric_only=True)["charges"].drop("charges").sort_values(ascending=False)
```

---

## Outlier Analysis

Outlier analysis was performed using boxplots.

The `bmi` and `charges` columns were visually inspected to identify possible extreme values. The BMI boxplot showed a few high-value observations, but these values were not removed because they are still realistic and medically possible.

The `charges` column also contained high insurance cost values. These observations were kept in the dataset because they may represent valid high-risk insurance cases, especially for smokers.

As a result, outliers were detected and interpreted, but they were not removed from the dataset.
---

## Data Preprocessing

Before training the models, categorical variables were converted into numerical values.

### Smoker Encoding

The `smoker` column was manually converted into binary values:

```text
no  -> 0
yes -> 1
```

Example:

```python
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
```

### One-Hot Encoding

The `sex` and `region` columns were encoded using `OneHotEncoder`.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ("encoder", OneHotEncoder(), ["sex", "region"])
    ],
    remainder="passthrough"
)

X = ct.fit_transform(X)
```

One-hot encoding was used to avoid creating an artificial numerical order between categories such as gender and region.

---

## Train-Test Split

The dataset was split into training and test sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=1
)
```

The training set was used to train the models, and the test set was used to evaluate model performance.

---

## Models Used

### 1. Multiple Linear Regression

Multiple Linear Regression was used as the baseline model.

It is simple and easy to interpret, but it assumes a mostly linear relationship between the input features and the target variable. Since medical insurance charges may depend on non-linear relationships, this model performed worse than the other models.

---

### 2. Random Forest Regressor

Random Forest Regressor was used to capture non-linear relationships in the data.

It performed much better than Multiple Linear Regression. Random Forest can also provide feature importance, which helps explain which variables have the strongest effect on the prediction.

---

### 3. Support Vector Regression

Support Vector Regression with RBF kernel was also tested.

Since SVR is sensitive to feature scale, StandardScaler was applied to both the input features and the target variable.

The predicted values were transformed back to the original scale using inverse transformation.

Example SVR preprocessing:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sc_X = StandardScaler()
sc_y = StandardScaler()

X_train_SVR = sc_X.fit_transform(X_train)
X_test_SVR = sc_X.transform(X_test)

y_train_SVR = sc_y.fit_transform(y_train.values.reshape(-1, 1))

regressor_svr = SVR(kernel="rbf")
regressor_svr.fit(X_train_SVR, y_train_SVR.ravel())

y_pred_svr_scaled = regressor_svr.predict(X_test_SVR)
y_pred_svr = sc_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1))
```

---

## Model Evaluation Metrics

The models were evaluated using the following regression metrics:

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error. Shows the average absolute prediction error. |
| MSE | Mean Squared Error. Penalizes larger errors more strongly. |
| RMSE | Root Mean Squared Error. Shows prediction error in the original target scale. |
| R² Score | Shows how well the model explains the variance in the target variable. |

Lower MAE, MSE, and RMSE values indicate better model performance.

A higher R² Score indicates better explanatory power.

---

## Model Results

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Multiple Linear Regression | 4376.26 | 43104275.10 | 6565.38 | 0.7272 |
| Random Forest Regressor | 2732.00 | 24010789.68 | 4900.08 | 0.8481 |
| Support Vector Regression | 2629.21 | 24170113.17 | 4916.31 | 0.8470 |

---

## Result Interpretation

Multiple Linear Regression had the weakest performance among the tested models. This suggests that the relationship between the input features and insurance charges is not fully linear.

Random Forest Regressor and Support Vector Regression produced very similar results.

SVR achieved the lowest MAE, which means it had the smallest average absolute prediction error. However, Random Forest achieved a slightly lower RMSE and a slightly higher R² Score.

This means Random Forest handled larger errors slightly better and explained the variance in insurance charges slightly more effectively.

Because the differences between Random Forest and SVR were very small, Random Forest was selected as the final model due to its interpretability and ability to provide feature importance.

---

## Final Model

The final selected model is:

```text
Random Forest Regressor
```

Reasons for selecting Random Forest:

- It achieved the highest R² Score.
- It achieved the lowest RMSE.
- It performed much better than Multiple Linear Regression.
- It can capture non-linear relationships.
- It provides feature importance.
- It is easier to interpret than SVR.

---

## Actual vs Predicted Visualization

Actual vs predicted plots were used to visually evaluate model performance.

Example for SVR:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred_svr.ravel(), alpha=0.7)

min_val = min(y_test.min(), y_pred_svr.min())
max_val = max(y_test.max(), y_pred_svr.max())

plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("SVR: Actual vs Predicted Charges")
plt.show()
```

In this plot, predictions closer to the diagonal line represent better model performance.

---

## Example Prediction

A new person’s information can be passed into the trained model to predict estimated insurance charges.

Example input:

| Feature | Value |
|---|---|
| age | 35 |
| sex | female |
| bmi | 23.8 |
| children | 0 |
| smoker | no |
| region | southwest |

Example prediction code:

```python
import pandas as pd

new_person = pd.DataFrame(
    [[35, "female", 23.8, 0, 0, "southwest"]],
    columns=["age", "sex", "bmi", "children", "smoker", "region"]
)

new_person_encoded = ct.transform(new_person)

prediction = regressor.predict(new_person_encoded)

print("Predicted insurance cost:", prediction[0])
```

---

## Key Findings

The main findings from this project are:

- Smoking status is one of the strongest predictors of insurance charges.
- Smokers generally have much higher insurance costs.
- Age and BMI also affect insurance charges.
- Random Forest and SVR performed much better than Multiple Linear Regression.
- Random Forest was selected as the final model because it provided strong predictive performance and better interpretability.

---

## Future Improvements

Possible improvements for this project include:

- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Cross-validation for more reliable model evaluation
- Testing additional models such as Gradient Boosting or XGBoost
- Applying log transformation to the target variable
- Creating a simple web application for user-based predictions
- Saving the final model using joblib or pickle

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/medical-insurance-cost-prediction.git
```

Go to the project directory:

```bash
cd medical-insurance-cost-prediction
```

Install the required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn jupyter
```

Open Jupyter Notebook:

```bash
jupyter notebook
```

Then open the notebook file:

```text
medical_insurance_cost_prediction.ipynb
```

Run the notebook cells from top to bottom.

---

## Project Structure

```text
medical-insurance-cost-prediction/
│
├── insurance.csv
├── medical_insurance_cost_prediction.ipynb
└── README.md
```

---

## Conclusion

This project demonstrates how regression models can be used to predict medical insurance charges.

The comparison between Multiple Linear Regression, Random Forest Regressor, and Support Vector Regression showed that non-linear models performed better on this dataset.

Random Forest Regressor was selected as the final model because it achieved strong prediction performance and provided better interpretability through feature importance.

Overall, this project shows the importance of data preprocessing, model comparison, and evaluation metrics in building a machine learning regression project.
