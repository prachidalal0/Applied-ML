# Customer Churn Prediction Using Machine Learning

## Class
BUAN 6341: Applied Machine Learning

## Date
Fall 2023

---

## Description
This project focuses on predicting customer churn using applied machine learning techniques. The objective is to build a robust classification model to identify customers at risk of leaving a subscription-based service. The dataset used contains detailed customer information, such as account age, monthly charges, subscription type, and churn status.

The project incorporates advanced preprocessing, feature engineering, and model evaluation techniques to optimize predictive performance. Several models, including Logistic Regression, Decision Trees, Random Forest, and ensemble methods like Bagging and AdaBoost, are trained and compared. Metrics such as accuracy, precision, recall, F1 score, and AUC-ROC are used to assess model performance.

### Key Highlights
- Handling class imbalance using weighted loss functions and hyperparameter tuning.
- Exploring feature relationships through visualization and statistical methods.
- Deploying scalable and interpretable models with L2 and L1 regularization.
- Incorporating ensemble methods and boosting techniques to enhance model accuracy and robustness.

---

## Key Features

### 1. Data Preprocessing and Feature Engineering
- **Missing Value Handling**: Verified no missing values in the dataset.
- **Feature Scaling**: Standardized numerical features using `StandardScaler`.
- **Feature Encoding**: Used one-hot encoding for categorical variables such as `SubscriptionType`, `PaymentMethod`, and `GenrePreference`.
- **Feature Selection**: Selected key predictors influencing churn based on domain knowledge and statistical analysis.

### 2. Machine Learning Models
- **Logistic Regression**: Baseline model with ridge (L2) and lasso (L1) regularization.
- **Decision Trees**: Simple interpretable models to identify important splits.
- **Random Forest**: An ensemble method to reduce overfitting and improve accuracy.
- **AdaBoost**: Boosting technique for improving weak learners.
- **Bagging Classifier**: Ensemble approach combining multiple Random Forest models.

### 3. Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **AUC-ROC Curve**

### 4. Visualization and Insights
- Explored feature distributions and relationships using histograms and pair plots.
- Analyzed churn trends, such as:
  - Snapshot churn rate.
  - Average account age for churned and retained customers.
  - Revenue impact of churn.

---

## Technologies Used
- **Scikit-learn**: For model training, evaluation, and hyperparameter tuning.
- **Pandas & NumPy**: For data manipulation and analysis.
- **Matplotlib**: For visualization of feature distributions and model performance.
- **Jupyter Notebook**: For interactive data exploration and experimentation.

---

## Methods Used

### Model Training
- **Baseline Models**: Trained Logistic Regression models with and without regularization.
- **Ensemble Techniques**: Leveraged Random Forest and Bagging Classifiers for improved stability.
- **Boosting**: Applied AdaBoost to enhance weak learners.
- **Hyperparameter Tuning**: Performed grid search for optimal `k` in KNN and hyperparameters in SVM.

### Data Preprocessing
- **Tokenization and Encoding**: Applied one-hot encoding for categorical features.
- **Standardization**: Scaled numerical data to improve model convergence.
- **Class Balancing**: Used weighted loss functions to handle class imbalance.

### Evaluation
- **Performance Metrics**: Assessed models using precision-recall and ROC curves.
- **Confusion Matrix Analysis**: Examined false positives and false negatives to improve decision-making thresholds.

### Visualization
- Precision-Recall and ROC Curves to evaluate trade-offs in model performance.
- Distribution plots to understand customer attributes.

---

## Future Work
- Incorporate additional feature engineering, such as interaction terms and polynomial features.
- Experiment with deep learning models for improved prediction accuracy.
- Deploy the model using AWS for real-time churn prediction.
- Implement Explainable AI techniques for better model interpretability.

---

## Acknowledgments
This project was developed as part of BUAN 6341: Applied Machine Learning (Fall 2023). Special thanks to the instructor for guidance and resources provided during the course.
