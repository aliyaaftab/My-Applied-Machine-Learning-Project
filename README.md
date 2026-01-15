Excited to share that i have recently completed a project focused on Machine Learning. The goal was to build and compare basic ML models using different hypotheses to analyse performance through feature selection and correlation.

Steps I followed:
1. Loaded and explored the dataset to identify the target variable (Time Taken).
2. Handled null values and performed categorical encoding.
3. Split data into training (80%) and testing (20%) sets.
4. Evaluated models using RMSE and R² score.

Code snippet:
# Train & evaluate function
def train_evaluate(model, X_train, X_test, y_train, y_test):
 model.fit(X_train, y_train)
 y_pred = model.predict(X_test)
 rmse = mean_squared_error(y_test, y_pred, squared=False)
 r2 = r2_score(y_test, y_pred)
 return rmse, r2

# Hypothesis 3 - After removing highly correlated features
model = LinearRegression()
rmse, r2 = train_evaluate(model, X_train, X_test, y_train, y_test)
print(f"R² Score: {r2:.3f}")

Results Summary:
 1. All variables → R² = 0.637
 2. Top correlated variables → R² = 0.649
 3. After removing highly correlated features → R² = 0.65 

Conclusion:
The third hypothesis performed best, highlighting the impact of feature selection and multicollinearity reduction in improving model accuracy.
