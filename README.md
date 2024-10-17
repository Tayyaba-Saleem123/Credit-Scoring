# **Credit Scoring Model Using Classification Algorithms**

This project aims to develop a **credit scoring model** that predicts the **creditworthiness of individuals** based on historical financial data. Various **classification algorithms** are utilized to assess the likelihood that an individual will repay their debt, and the model’s accuracy is thoroughly evaluated.

## **Key Features of This Project:**

- **Data Collection and Preprocessing**:  
  Financial data such as age, job, income, savings accounts, checking accounts, loan history, and credit usage are cleaned and prepared for analysis. Missing values are imputed, and categorical variables are encoded to transform the data into a format suitable for machine learning models.

- **Algorithm Selection**:  
  Multiple classification algorithms are considered for this task, including:
  - **Random Forest Classifier**
  - **Logistic Regression**
  - **Support Vector Machines (SVM)**
  - **Decision Trees**

  Each algorithm is evaluated to determine which one yields the best results for credit risk prediction.

- **Model Training and Validation**:  
  The dataset is split into training and test sets to allow for the accurate evaluation of each model. Cross-validation is used to further ensure the model’s robustness and avoid overfitting.

- **Performance Metrics**:  
  The model’s performance is assessed using key metrics such as:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1 Score**
  - **Confusion Matrix**
  
  These metrics help in determining the model’s ability to correctly classify individuals as either creditworthy or not.

- **Class Imbalance Handling**:  
  Since credit datasets often suffer from class imbalance (more "Good" credit risks than "Bad"), techniques such as **class weighting** and **SMOTE** (Synthetic Minority Over-sampling Technique) are implemented to balance the dataset.

- **Prediction Functionality**:  
  After training, the model is capable of predicting the creditworthiness of new individuals based on their financial data. This functionality allows lenders to make informed decisions on whether to approve loans or extend credit.

- **Model Comparison**:  
  Different models are compared based on their performance metrics, and the best model is selected for deployment. Feature importance is also analyzed to understand which financial factors play the biggest role in predicting creditworthiness.

## **Additional Resources**

- **Output Image**: A sample output image is provided to illustrate the model’s prediction results and performance.

![Output](https://github.com/user-attachments/assets/193556d4-5ae1-4042-8b5a-e703dd79bf9a)

- **Video Walkthrough**: A comprehensive video walkthrough of the entire project, from data preprocessing to model evaluation, is included to help users understand the process step-by-step.

https://github.com/user-attachments/assets/58c0b277-02eb-4736-aa8b-c7a86c9f6f7c

## **Conclusion**:
This credit scoring model demonstrates how machine learning can be effectively applied to **financial risk assessment**, providing a robust solution for predicting individual creditworthiness. The model helps financial institutions make better decisions when approving loans or extending credit, reducing the risk of defaults and improving financial stability.
