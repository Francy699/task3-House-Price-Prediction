
# Linear Regression on Housing Price Dataset 🏡

This project implements **Linear Regression** using Python and scikit-learn to predict housing prices based on features like area, number of bedrooms, and bathrooms.

---

## 📚 What is Linear Regression?

**Linear Regression** is a supervised machine learning algorithm used to predict a continuous target variable based on one or more input features.

![Linear Regression](https://github.com/user-attachments/assets/37557cc7-19aa-4e8f-b816-25825e046829)

- It tries to find the **best-fitting straight line** (in 2D) or hyperplane (in higher dimensions) that minimizes the error between predicted and actual values.
- **Equation of Linear Regression:**
  \[
  y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \ldots + \beta_nx_n
  \]
  where:
  - \( y \) = predicted value
  - \( \beta_0 \) = intercept
  - \( \beta_1, \beta_2, \ldots, \beta_n \) = coefficients (slopes)
  - \( x_1, x_2, \ldots, x_n \) = feature values

The goal during training is to **learn** the best \(\beta\) values that minimize the **Mean Squared Error (MSE)**.

---

## 🛠️ Complete Code Explanation

### 1. Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
- **pandas**: Data manipulation and analysis.
- **numpy**: Mathematical operations.
- **matplotlib.pyplot**: Data visualization.

---

### 2. Load the Dataset
```python
df = pd.read_csv('/content/Housing.csv')
```
- Load the CSV file containing the housing data.

---

### 3. View Dataset
```python
df.head()
```
- Display the first five rows of the dataset to understand its structure.


!(https://github.com/user-attachments/assets/024b9b2e-c776-45b5-a0c2-7f6ac8805fb2)

---

### 4. Statistical Summary
```python
df.describe()
```
- View basic statistical details like count, mean, std deviation, min, and max values for numerical columns.

---

### 5. Dataset Info
```python
df.info()
```
- Get information about dataset columns, data types, and missing values.

---

### 6. Dataset Shape
```python
print('Shape')
print(df.shape)
```
- Display the number of rows and columns.

---

### 7. List of Column Names
```python
df.columns
```
- Print all column names.

![Screenshot 2025-04-28 165517](https://github.com/user-attachments/assets/6df805eb-2f77-4906-b9ea-5640542e7ad6)


---

### 8. Feature and Target Variable
```python
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']
```
- **X**: Features (area, bedrooms, bathrooms)
- **y**: Target (price)

---

### 9. Splitting the Dataset
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
```
- Split data into training (60%) and testing (40%) sets.

---

### 10. Build and Train the Model
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```
- Create a **LinearRegression** object and train it on training data.

![Screenshot 2025-04-28 165531](https://github.com/user-attachments/assets/bd2b3660-d519-43c5-876f-824624a094bc)

---

### 11. Model Intercept
```python
print(model.intercept_)
```
- Intercept \( \beta_0 \) is printed.

---

### 12. Model Coefficients
```python
df_coeff = pd.DataFrame(model.coef_.reshape(-1, 1), index=X.columns, columns=['Coefficient'])
print(df_coeff)
```
- Print the learned coefficients for each feature.

---

### 13. Make Predictions
```python
y_pred = model.predict(X_test)
```
- Predict house prices using the test set.

---

### 14. Visualize Predictions
```python
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
```
- Scatter plot comparing actual vs predicted prices.

![Screenshot 2025-04-28 165548](https://github.com/user-attachments/assets/cd264cf8-b544-49e3-9829-3158cdd3e602)



---

### 15. Model Evaluation
```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Mean Squared Error:', mse)
print('R2 Score:', r2)
```
- Evaluate model performance using Mean Squared Error (MSE) and R² score.

---

### 16. Visualize Coefficients
```python
plt.figure(figsize=(8,6))
plt.bar(df_coeff.index, df_coeff['Coefficient'], color='skyblue')
plt.title('Feature Coefficients', fontsize=20)
plt.xlabel('Features', fontsize=16)
plt.ylabel('Coefficient Value', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```
- Visualizes how much each feature contributes to the target price.

---

## 🎯 Summary of Results

- **Bathrooms** had the highest impact on the price.
- The model achieved an **R² score of ~0.43**, meaning it explains about 43% of the variability in house prices.
- Future improvements could involve using more features and applying advanced techniques like regularization.

---

# 📌 Author

> Developed by a Machine Learning Enthusiast 🚀

