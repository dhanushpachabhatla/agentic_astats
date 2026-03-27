# Analysis Plan

## User Goal
Perform exploratory data analysis (EDA) and build a classification model to predict iris species.

## Dataset Summary
The `iris.csv` dataset comprises 150 observations of iris flowers, detailing four continuous measurements (sepal length, sepal width, petal length, petal width) and their corresponding categorical species (setosa, versicolor, virginica). All columns are complete with no missing values, and each species is equally represented with 50 samples.

## Analysis Steps

- **Step 1: Data Preprocessing - Target Encoding**
  - **Objective:** Convert the categorical 'species' column into a numerical format suitable for machine learning algorithms.
  - **Method:** Apply Label Encoding to the `species` column.
  - **Expected Output:** A dataset with the `species` column encoded numerically (e.g., 0, 1, 2 for setosa, versicolor, virginica).
  - **Priority:** High

- **Step 2: Univariate Exploratory Data Analysis**
  - **Objective:** Understand the distribution, central tendency, and spread of each numerical feature, and identify potential outliers or skewness.
  - **Method:**
    - Generate histograms for `sepal_length`, `sepal_width`, `petal_length`, and `petal_width` to visualize their distributions.
    - Create box plots for each numerical feature to identify outliers and quartiles.
    - Calculate and review descriptive statistics (mean, median, standard deviation, min, max, skewness, kurtosis) for all numerical columns.
  - **Expected Output:** Histograms, box plots, and a summary table of descriptive statistics for each numerical feature.
  - **Priority:** High

- **Step 3: Bivariate and Multivariate EDA for Species Differentiation**
  - **Objective:** Investigate relationships between numerical features and how they differ across the three iris species to understand which features best separate the classes.
  - **Method:**
    - Generate scatter plots for all pairs of numerical features, colored by `species`, using `seaborn.pairplot`.
    - Calculate and present grouped descriptive statistics (mean, standard deviation) for each numerical feature, broken down by `species`.
    - Perform One-way ANOVA tests for each numerical feature to assess if there are significant differences in their means across the three species groups, as allowed by statistical constraints.
  - **Expected Output:** Pairplot visualization, tables of grouped statistics, ANOVA test results (F-statistic, p-value) for each numerical feature by species.
  - **Priority:** High

- **Step 4: Data Splitting for Classification**
  - **Objective:** Divide the dataset into training and testing sets to prepare for model development and evaluation, ensuring the model's performance can be assessed on unseen data.
  - **Method:** Split the dataset into features (X) and target (y), then use `sklearn.model_selection.train_test_split` to create training and testing subsets (e.g., 80% train, 20% test), stratifying by `species` to maintain class proportions.
  - **Expected Output:** `X_train`, `X_test`, `y_train`, `y_test` datasets.
  - **Priority:** High

- **Step 5: Species Classification Model Training**
  - **Objective:** Develop a machine learning model capable of classifying iris species based on sepal and petal measurements.
  - **Method:** Train a Logistic Regression classifier on the `X_train` and `y_train` datasets.
  - **Expected Output:** A trained Logistic Regression model.
  - **Priority:** High

- **Step 6: Model Evaluation and Interpretation**
  - **Objective:** Assess the performance of the trained classification model and interpret its predictions.
  - **Method:**
    - Use the trained model to make predictions on the `X_test` dataset.
    - Calculate standard classification metrics: accuracy, precision, recall, F1-score.
    - Generate and analyze a confusion matrix to understand true positive, true negative, false positive, and false negative rates for each species.
    - Examine the model coefficients (for Logistic Regression) to infer the importance and direction of influence of each feature on species classification.
  - **Expected Output:** Classification report (metrics), confusion matrix, and interpretation of feature importance from model coefficients.
  - **Priority:** High

## Notes
- The "high cardinality" warning in the dataset profile for the `species` column is noted but will be disregarded as `species` only has 3 unique values, which is low cardinality.
- The analysis plan assumes that general machine learning classification algorithms (like Logistic Regression) are permissible, as the "forbidden methods" in the statistical constraints primarily refer to specific statistical inference tests rather than predictive modeling techniques.
- Although outlier detection was recommended in the dataset profile, given the classic nature of the Iris dataset and its common use in introductory machine learning, robust outlier handling is not prioritized unless explicitly observed to be problematic during EDA.
- The independent observations assumption (cross-sectional data) is satisfied and underlies the chosen statistical methods.