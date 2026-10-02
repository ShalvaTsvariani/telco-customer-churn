# Telco Customer Churn Prediction

A machine learning project to predict customer churn and identify the factors associated with customers leaving a telecommunications company.

The project uses the **Telco Customer Churn dataset**, containing 7,043 customers and 21 original features.

The main objective is not only to build a predictive model, but also to evaluate it carefully and translate the results into interpretable customer-retention insights.

## Project Overview

The notebook follows an end-to-end machine learning workflow:

**Data Cleaning → Exploratory Data Analysis → Feature Engineering → Train/Test Split → Preprocessing Pipeline → Cross-Validation → Model Comparison → Hyperparameter Tuning → Test Evaluation → Threshold Analysis → Model Interpretation**

Three classification models were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Although the tree-based models were more complex, all three achieved very similar cross-validation performance. Logistic Regression was therefore selected as the final model because it provided comparable predictive performance while remaining easier to interpret.

## Dataset

The dataset contains information about customers' demographics, services, contract information, payment methods, and charges.

### Dataset size

* **7,043 customers**
* **21 original features**
* Target variable: `Churn`

The target is binary:

* `0` → Customer did not churn
* `1` → Customer churned

The dataset is imbalanced, with approximately **26.5% of customers being churners**. Because of this, accuracy alone is not an appropriate metric for evaluating the model.

## Exploratory Data Analysis

The analysis examined churn rates across categorical and numerical variables.

### Main findings

**Contract type**

Contract type showed the largest difference in churn rates:

* Month-to-month: **42.7%**
* One year: **11.3%**
* Two year: **2.8%**

**Internet service**

* Fiber optic: **41.9%**
* DSL: **19.0%**
* No internet service: **7.4%**

**Payment method**

Customers using electronic checks had a churn rate of approximately **45.3%**, substantially higher than the other payment methods.

**Tenure**

Short-tenure customers were considerably more likely to churn. Churn decreased sharply as customers remained with the company longer.

The first six months represented the highest-risk period, with a churn rate of approximately **52.9%**.

**Monthly charges**

Churners generally had higher monthly charges. Their median monthly charge was approximately **$80**, compared with approximately **$64** for non-churners.

### Correlation analysis

`tenure` had the strongest linear correlation with churn at approximately **-0.35**.

There was also substantial correlation between:

* `tenure` and `TotalCharges`: **0.83**
* `MonthlyCharges` and `TotalCharges`: **0.65**

This is expected because total charges are strongly related to how long a customer has stayed and their monthly charge.

## Feature Engineering

Two additional features were created:

### `TotalServices`

Counts the number of services used by each customer, including:

* Phone service
* Multiple lines
* Internet service
* Online security
* Online backup
* Device protection
* Tech support
* Streaming TV
* Streaming movies

The relationship between service count and churn was non-linear.

### `TenureGroup`

Customers were grouped into tenure intervals:

* 0–6 months
* 7–12 months
* 13–24 months
* 25–48 months
* 49–60 months
* 61–72 months

This feature helped capture the strong relationship between early customer tenure and churn.

## Data Preprocessing

The preprocessing pipeline was implemented using `ColumnTransformer` and `Pipeline`.

### Categorical features

Categorical variables were transformed using:

```python
OneHotEncoder(
    drop='first',
    handle_unknown='ignore'
)
```

### Numerical features

The numerical features:

* `tenure`
* `MonthlyCharges`
* `TotalCharges`

were standardized using `StandardScaler`.

Preprocessing was kept inside the modeling pipeline so that transformations were fitted only on the training portion of each cross-validation fold, preventing data leakage.

## Train/Test Split

The dataset was split into:

* **80% training**
* **20% test**

A stratified split was used to preserve the churn/non-churn class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The test set was kept untouched during model selection and hyperparameter tuning.

## Model Comparison

All models were evaluated using **5-fold Stratified Cross-Validation**.

The following metrics were measured:

* Accuracy
* Precision
* Recall
* F1
* ROC-AUC

### Cross-validation results

| Model               | Accuracy | Precision | Recall |    F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | ----: | ------: |
| Logistic Regression |    0.751 |     0.520 |  0.795 | 0.628 |   0.848 |
| Random Forest       |    0.757 |     0.530 |  0.774 | 0.629 |   0.847 |
| XGBoost             |    0.749 |     0.518 |  0.804 | 0.630 |   0.847 |

The three models performed very similarly.

This suggests that increasing model complexity did not provide a meaningful improvement with the available features.

Logistic Regression was selected as the final model because it achieved comparable predictive performance while providing a more interpretable relationship between features and churn probability.

## Hyperparameter Tuning

`GridSearchCV` was used to tune the Logistic Regression model.

The following parameters were evaluated:

* Regularization strength `C`
* `class_weight`

The search used the same stratified 5-fold cross-validation strategy and was performed exclusively on the training data.

### Best parameters

```text
C = 0.03
class_weight = 'balanced'
```

The best cross-validated F1 score was:

```text
F1 = 0.632
```

The improvement over the baseline Logistic Regression was small:

| Metric    | Baseline | Tuned | Difference |
| --------- | -------: | ----: | ---------: |
| Accuracy  |    0.751 | 0.757 |     +0.007 |
| Precision |    0.520 | 0.529 |     +0.009 |
| Recall    |    0.795 | 0.786 |     -0.009 |
| F1        |    0.628 | 0.632 |     +0.004 |
| ROC-AUC   |    0.848 | 0.846 |     -0.001 |

The small improvement indicates that hyperparameter tuning had limited impact on the model's performance.

## Final Test Set Performance

After model selection and tuning, the final Logistic Regression model was evaluated once on the previously untouched test set.

| Metric    | Test Score |
| --------- | ---------: |
| Accuracy  |  **0.744** |
| Precision |  **0.512** |
| Recall    |  **0.786** |
| F1        |  **0.620** |
| ROC-AUC   |  **0.844** |

The confusion matrix was:

```text
[[755 280]
 [ 80 294]]
```

This means the model correctly identified:

* **294 of 374 churners**
* **755 of 1,035 non-churners**

The recall of **0.786** means the model identifies approximately **79% of actual churners**.

However, precision is **0.512**, meaning that approximately half of the customers flagged as potential churners actually churned.

This trade-off is expected when prioritizing the detection of churners through class weighting.

## Decision Threshold Analysis

The default classification threshold of `0.5` was also examined.

Rather than selecting a threshold using the test set, out-of-fold predictions from the training data were used to evaluate the trade-off.

| Threshold | Precision | Recall |    F1 | Customers Flagged |
| --------: | --------: | -----: | ----: | ----------------: |
|       0.3 |     0.420 |  0.924 | 0.578 |             58.3% |
|       0.4 |     0.473 |  0.863 | 0.611 |             48.4% |
|       0.5 |     0.529 |  0.786 | 0.632 |             39.5% |
|       0.6 |     0.581 |  0.690 | 0.631 |             31.5% |
|       0.7 |     0.653 |  0.567 | 0.607 |             23.0% |

Lowering the threshold increases recall but also increases the number of customers who would receive a retention intervention.

The appropriate threshold would ultimately depend on the business cost of:

* Missing a customer who is going to churn
* Contacting a customer who would not have churned

## Model Interpretation

Because the final model is Logistic Regression, its coefficients can be converted into odds ratios to understand the relationship between features and predicted churn.

The analysis identified several strong associations with churn.

### Contract type

Compared with month-to-month customers, longer-term contracts were associated with substantially lower churn odds.

### Tenure

Churn odds decreased as tenure increased, with the largest risk concentrated among newer customers.

### Fiber optic service

Fiber optic customers showed higher predicted churn odds compared with the reference category.

### Security and support services

Customers with services such as:

* Online Security
* Tech Support

showed lower churn odds.

### Streaming services

Streaming TV and Streaming Movies were associated with higher churn odds in the final model.

### Payment method and billing

Electronic-check users and customers using paperless billing showed higher churn odds.

### Important limitation

These relationships are **associations from a predictive model, not causal effects**.

For example, the lower churn associated with longer contracts does not prove that longer contracts themselves cause customers to remain. Customers who choose longer contracts may already differ from month-to-month customers in other ways.

## Key Takeaways

1. **Customer tenure is strongly associated with churn**, with the highest risk concentrated among new customers.
2. **Contract type is one of the strongest predictors of churn**, with substantially different churn rates across contract categories.
3. **Fiber optic customers show higher churn rates** than DSL and customers without internet service.
4. **Electronic-check customers have substantially higher churn rates** than customers using other payment methods.
5. **Security and support services are associated with lower churn**, while some streaming services are associated with higher churn.
6. **Model complexity did not substantially improve predictive performance**: Logistic Regression, Random Forest, and XGBoost achieved very similar cross-validation results.
7. The final model achieves **0.844 ROC-AUC and 0.786 recall** on the held-out test set.
8. **Threshold selection is a business decision** because increasing recall comes with more false positives.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

### Machine Learning Techniques

* Exploratory Data Analysis
* Feature Engineering
* One-Hot Encoding
* Feature Scaling
* Logistic Regression
* Random Forest
* XGBoost
* Stratified Train/Test Split
* Stratified K-Fold Cross-Validation
* Grid Search
* Classification Metrics
* ROC-AUC
* Decision Threshold Analysis
* Model Coefficient / Odds Ratio Interpretation

## Project Structure

```text
Telco-Customer-Churn/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── Telco_Churn.ipynb
│
└── README.md
```

## How to Run

Clone the repository and install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Then open the notebook:

```bash
jupyter notebook Telco_Churn.ipynb
```

Make sure the dataset is located in:

```text
data/Telco-Customer-Churn.csv
```

## Project Objective

This project was developed as part of my transition from **Data Analytics toward Data Science**, with a focus on developing practical experience in:

* Exploratory data analysis
* Feature engineering
* Classification
* Model evaluation
* Cross-validation
* Hyperparameter tuning
* Avoiding data leakage
* Model interpretation
* Translating machine learning results into business-relevant insights
