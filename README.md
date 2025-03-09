Crop Production Prediction Dashboard - Documentation

1. Introduction

The Crop Production Prediction Dashboard is a data-driven application designed to analyze agricultural datasets, visualize trends, and predict crop production based on historical data. This project leverages data preprocessing, exploratory data analysis (EDA), and machine learning techniques to provide actionable insights into agricultural productivity.

2. Approach

2.1 Data Collection

The dataset is sourced from FAOSTAT, containing records of crop production, area harvested, and yield across various countries and years.

The dataset is stored in an Excel file and loaded into the Streamlit application for processing.

2.2 Data Cleaning and Preprocessing

Column Standardization: Stripped unnecessary spaces and renamed columns for consistency.

Filtering Data: Retained only relevant elements – 'Area harvested', 'Yield', and 'Production'.

Pivoting Data: Transformed the dataset to have 'Area harvested', 'Yield', and 'Production' as separate columns.

Handling Missing Values: Filled missing values with 0 to ensure a complete dataset.

2.3 Exploratory Data Analysis (EDA)

EDA helps uncover patterns and trends in agricultural data through visualizations:

Production Trend Over Time: A line chart shows crop production variations over the years.

Outlier Detection: A boxplot highlights anomalies in production data.

Area vs. Production Relationship: A regression plot visualizes the correlation between area harvested and production.

2.4 Model Training and Evaluation

Model Selection: Linear Regression was chosen to predict crop production based on area harvested, yield, and year.

Train-Test Split: The dataset was split into 80% training and 20% testing sets.

Performance Metrics:

Mean Squared Error (MSE)

R-squared Score (R²)

Mean Absolute Error (MAE)

3. Key Findings

There is a strong correlation between area harvested and crop production, as expected.

Some countries exhibit significant fluctuations in production due to external factors (e.g., climate conditions, policies).

The linear regression model achieved an acceptable R² score, indicating its usefulness for production prediction.

Outliers detected in production trends suggest potential inconsistencies in reporting or external disruptions.

4. Actionable Insights

Agricultural Planning: Farmers and policymakers can use predictive insights to optimize resource allocation.

Risk Mitigation: Identifying production outliers helps in understanding and mitigating agricultural risks.

Improved Forecasting: The model can be extended with additional factors (e.g., climate, soil quality) to improve accuracy.

Data Quality Improvement: Addressing missing or inconsistent data can enhance future predictive performance.

5. Conclusion

This dashboard provides an interactive and data-driven approach to analyzing and predicting crop production. By leveraging historical agricultural data and machine learning, it serves as a valuable tool for stakeholders in the agriculture sector to make informed decisions.

