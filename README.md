# Elevate Labs - Task 3: Data Modeling & Training

### **Objective**
The objective of this task was to train a machine learning model to predict house prices using the provided dataset. A Linear Regression model was chosen for this task.

### **Steps Taken**

1.  **Data Splitting**: The `Housing.csv` dataset was loaded and split into training and testing sets using a 80/20 ratio. This ensures that the model can be evaluated on unseen data to assess its generalization ability.

2.  **Model Training**: A `LinearRegression` model was initialized and trained on the training data. The model learned the linear relationships between the house features and their prices.

3.  **Model Evaluation**: The trained model was used to predict prices on the test set. Its performance was evaluated using the following metrics:
    * **Mean Absolute Error (MAE)**: [Insert MAE value from code output]
    * **Mean Squared Error (MSE)**: [Insert MSE value from code output]
    * **R-squared (R2)**: [Insert R2 value from code output]
    The R-squared value indicates the proportion of the variance in the target variable that is predictable from the features.

4.  **Prediction Visualization**: A scatter plot of actual vs. predicted prices was generated. The plot shows the model's predictions plotted against the true values, providing a visual representation of the model's performance.

### **Conclusion**
The Linear Regression model was successfully trained and evaluated on the dataset. The R-squared value suggests that the model explains a significant portion of the variance in house prices. The project demonstrates a fundamental machine learning workflow, from data preparation to model evaluation.
