# Advanced Sports Bet Recommender (ASBR)

Welcome to the wild world of Advanced Sports Bet Recommender (ASBR)! This engine uses a variety of data analysis and machine learning techniques to predict outcomes of future golf tournaments based on historical data.

## Workflow

1. **Connect to Database**: Connects to a SQLite database that contains all the historical data of golf tournaments.

2. **Retrieve Data**: Retrieves all the data from the `raw_data` table in the database. This includes information like player names, scores, and other statistics from past tournaments.

3. **Data Preprocessing**: Processes the raw data to make it suitable for further analysis and modeling. This includes checking and filtering various columns, normalizing some columns, selecting important features, handling missing values, and reducing dimensionality.

4. **Feature Engineering**: Enhances the data by creating new features or modifying existing ones to improve the performance of the machine learning model.

5. **Model Development**: Develops a machine learning model that can predict the outcomes of future tournaments based on the processed data.

6. **Predict Future Tournaments**: Uses the developed model to predict the outcomes of upcoming tournaments.

7. **Evaluate Predictions**: Evaluates the predictions made by the model by comparing them with the actual outcomes.

## Detailed Steps

### Data Preprocessing

Data preprocessing is an essential step where we clean and transform the raw data to make it suitable for further analysis and modeling.

The steps include:

1. **Filtering**: Columns like 'round_score', 'sg_columns', 'round_num', etc., are checked and filtered based on certain conditions to make sure the data in them is valid and reliable.

2. **Normalization**: The 'sg' columns are normalized relative to field strength.

3. **Feature Selection**: Selects the features based on chi-square test for round scores.

4. **Encoding**: The 'tour' column is stripped of leading/trailing spaces. The 'course_name', 'course_par', and 'course_num' columns are encoded and normalized using One-Hot Encoding and Standard Scaler, respectively.

5. **Dimensionality Reduction**: The number of features in the data is reduced using Principal Component Analysis (PCA) to make the model more manageable and less prone to overfitting. The number of principal components is set to 7, but this can be adjusted as needed.

The preprocessing results in a dataset that is clean, normalized, selected, encoded, and dimensionally reduced.

### Feature Engineering

This is a placeholder for any feature engineering steps, such as creating new features or modifying existing ones, to improve the performance of the machine learning model.

### Model Development

This is a placeholder for developing a machine learning model that can predict the outcomes of future tournaments based on the processed data.

### Predict Future Tournaments

This is a placeholder for using the developed model to predict the outcomes of upcoming tournaments.

### Evaluate Predictions

This is a placeholder for evaluating the predictions made by the model by comparing them with the actual outcomes.

## Data Flow Diagram

Here is a simple data flow diagram illustrating the workflow:

```
[Raw Data] --> [Data Preprocessing] --> [Feature Engineering] --> [Model Development] --> [Predict Future Tournaments] --> [Evaluate Predictions]
```

The raw data is first processed and transformed, after which new features may be engineered. The transformed data is then used to develop a predictive model. The model is used to predict future tournaments, and these predictions are evaluated against the actual outcomes.

Remember, this is a working script and it's still being developed. This means that some parts are placeholders waiting to be filled with the appropriate code for each step.