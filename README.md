# IBM-Attrition-Prediction-Model
Machine learning models used to predict the attrition of employees in IBM
1. Analysis
1.1 Research Questions
What kind of people tend to have a higher salary? What factors have the greatest impact on the monthly income? Can we predict the monthly income based on other features of the employees?
Which factors have the greatest impact on attrition? Can we predict the attrition of employees based on other characteristics?

1.2 Exploratory Data Analysis
1.2.1 Dataset Description
The dataset used in this project is the "IBM HR Analytics Employee Attrition & Performance" dataset obtained from the Kaggle website (https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset?select=WA_Fn-UseC_-HR-Employee-Attrition.csv). It consists of 1470 rows and 35 columns, with 9 numerical variables and 26 categorical variables. Each row represents the attributes of an employee from IBM.

1.2.2 Data Preparation
The dataset was checked for missing data, empty or duplicated records, and outliers. Outliers were removed to improve the prediction accuracy. Three variables ('Over18', 'EmployeeCount', and 'EmployeeNumber') were also removed as they were deemed irrelevant for predictions. Categorical variables were encoded, with binary encoding used for variables like 'Gender' and one-hot encoding for variables like 'EducationField'.

1.2.3 Analysis and Interpretation
After data cleaning, the dataset consisted of 46 numeric variables and 1435 records. Correlation analysis revealed that variables such as 'JobLevel' and 'TotalWorkingYears' had the highest correlation with monthly income. The relationship between monthly income and these variables was visualized through distribution plots.

1.3 Random Forest for Predicting Monthly Income
1.3.1 Method and Reasons
Random Forest regression was chosen to predict monthly income due to its high accuracy, scalability, and tolerance to missing data and outliers.

1.3.2 Analysis Steps
The dataset was randomly divided into training and test sets with a 7:3 ratio. The randomForest function was used to build a random forest model, and the % Var explained metric was used to assess the model's overall explained variance. The model's predictive performance was evaluated using the R2 metric and visualized through scatter plots. The importance of independent variables was determined based on the %IncMSE and IncNodePurity metrics.

1.3.3 Results
The top 6 important variables for predicting monthly income were identified as 'JobLevel', 'TotalWorkingYears', 'JobRoleManager', 'JobRoleResearch.Director', 'YearsAtCompany', and 'Age'. These variables explained about 93.94% of the total variance, indicating a good fit of the model. The optimized random forest model demonstrated high accuracy and closely predicted the actual monthly income.

1.4 SVM & Logistic Regression for Predicting Attrition
1.4.1 SVM
1.4.1.1 Method and Reasons
Support Vector Machines (SVM) were chosen for predicting attrition due to their ability to handle data with irregular distribution patterns and unknown distribution.

1.4.1.2 Analysis Steps
Similar to the random forest analysis, the dataset was divided into training and test sets. The svm function was used to build the SVM model, and the tune.svm function was used for tuning the model parameters. The performance of the optimized model was evaluated using the confusion matrix and ROC curve.

1.4.2 Logistic Regression
1.4.2.1 Method and Reasons
Logistic regression was selected for predicting attrition due to its interpretability and ability to model binary outcomes.

1.4.2.2 Analysis Steps
The dataset was divided into training and test sets, similar to the previous analyses. The glm function with a binomial family was used to build the logistic regression model. Model performance was evaluated using metrics such as accuracy, precision, recall, and the ROC curve.

1.4.2.3 Results
The logistic regression model achieved an accuracy of 86.5% in predicting attrition. The model's precision, recall, and F1-score were also calculated, indicating its ability to correctly identify employees who are likely to leave the company. The ROC curve demonstrated the model's overall performance in distinguishing between employees who will or will not churn.

2. Conclusion
This project aimed to analyze the factors influencing monthly income and employee attrition in an IBM dataset. The random forest regression model revealed that variables such as job level, total working years, and age had the greatest impact on monthly income. The SVM and logistic regression models successfully predicted employee attrition, with the logistic regression model achieving an accuracy of 86.5%.
Overall, this analysis provides valuable insights into the factors affecting employee income and attrition, which can aid in making data-driven decisions to retain employees and optimize compensation strategies.

3. Future Work
There are several avenues for future work based on this project:
Explore other machine learning algorithms, such as gradient boosting or neural networks, to compare their performance in predicting monthly income and attrition.
Conduct feature engineering to create new variables that might better capture the relationship between the predictors and the target variables.
Perform a more in-depth analysis of specific job roles or departments within the company to understand their unique factors influencing income and attrition.
Collect additional data, such as performance ratings or employee satisfaction surveys, to enhance the predictive models and gain more comprehensive insights.

4. Acknowledgements
The analysis in this project was conducted using the "IBM HR Analytics Employee Attrition & Performance" dataset obtained from Kaggle. The dataset was originally compiled by Pavan Subhash and is available at https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset. We acknowledge the contributions of Pavan Subhash and the Kaggle community in providing this dataset for analysis purposes.
