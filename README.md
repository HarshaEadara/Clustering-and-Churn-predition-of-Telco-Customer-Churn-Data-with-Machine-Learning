# Clustering-and-Churn-predition-of-Telco-Customer-Churn-Data-with-Machine-Learning
This repository contains a comprehensive Jupyter Notebook that showcases the application of advanced clustering techniques and machine learning models to analyze and predict customer churn in the telecommunications industry. It offers practical insights and actionable results that can be directly applied to enhance customer retention strategies and operational efficiency.

## Table of Contents
- [Overview](#overview)
- [Key Objectives](#key-objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Work Flow](#work-flow)
- [Results and Evaluation](#results-and-evaluation)
- [Usage](#usage)
- [Contributing](#contributing)

## Overview
Customer churn is a critical issue for telecom companies, as retaining existing customers is often more cost-effective than acquiring new ones. This project analyzes telco customer churn data, segments customers using clustering techniques, and predicts churn using various machine learning algorithms. The analysis provides actionable insights to develop targeted retention strategies by identifying high-risk customers and understanding key factors contributing to churn. Additionally, it highlights patterns that can inform proactive measures to minimize churn.
- Provides a framework for analyzing customer behavior and identifying churn risks early.
- Demonstrates how clustering can uncover hidden customer segments for targeted strategies.
- Offers comparative insights into model performance to select the best approach for churn prediction.

## Key Objectives

The main objectives of this project are:
- Analyze the Telco Customer Churn dataset to understand customer behavior and identify key churn drivers.
- Apply clustering techniques to segment customers and discover actionable patterns.
- Build and evaluate machine learning models to predict churn with high accuracy and reliability.
- Provide insights to aid telecom companies in implementing effective customer retention strategies.
  
## Dataset
The Telco Customer Churn dataset provides comprehensive information on customers, including their demographics, service details, and churn status. Key features include:

- **Customer ID:** Unique identifier for each customer.
- **Demographics:** Gender, age group, and other personal details.
- **Service Details:** Contract type, tenure, internet services, etc.
- **Churn:** Indicates whether a customer has churned or not.

You can find the **Telco Customer Churn** dataset in the `data` folder of this repository. If you prefer to download it yourself, ensure you get the dataset from Kaggle and place it in the `data` directory.

> **Note**: This dataset contains anonymized transaction data, making it ideal for training machine learning models while ensuring customer privacy.

### Preprocessing Steps
- Removal of duplicate entries
- Filtering of sparse user-product interactions
- Normalization and preparation for models

## Technologies Used
The project is implemented using:
- **Programming Language:** Python
- **Libraries and Frameworks:**
   - scikit-learn
   - Pandas
   - Matplotlib
   - Seaborn
   - NumPy
- **Jupyter Notebook:** For analysis and visualization

## Work Flow
The notebook follows the steps below:

1. **Importing Libraries and Data**
2. **Exploratory Data Analysis (EDA):** Analyzing feature distributions and identifying relationships with churn.
3. **Customer Segmentation with K-Means Clustering:**
   - Clustering customers based on attributes to identify distinct segments.
4. **Machine Learning Models for Churn Prediction:**
   - **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
   - **K-Nearest Neighbors (KNN)**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **Logistic Regression**
5. **Hyperparameter Tuning:** Optimizing model parameters for improved performance.
6. **Analyzing Prediction Scores:** Evaluating models using metrics like accuracy, F1-score, and Jaccard score.

## Results and Evaluation
### Performance of Models
The performance of the models was evaluated based on key metrics, including Accuracy Score, F1 Score, and Jaccard Score. Here are the detailed results:
- **K-Nearest Neighbors (KNN)**
   - Accuracy Score: 0.767
   - F1 Score: 0.758
   - Jaccard Score: 0.735

The K-Nearest Neighbors model shows decent performance with an accuracy of 76.7%. The F1 Score of 75.8% and Jaccard Score of 73.5% indicate that the model balances precision and recall effectively but is slightly less reliable compared to the other models.

- **Decision Tree Classifier**
   - Accuracy Score: 0.779
   - F1 Score: 0.772
   - Jaccard Score: 0.744

The Decision Tree classifier performs better with an accuracy of 77.9%. Its F1 Score of 77.2% and Jaccard Score of 74.4% reflect its ability to make more reliable predictions and distinguish between classes effectively.

- **Random Forest Classifier**
   - Accuracy Score: 0.789
   - F1 Score: 0.779
   - Jaccard Score: 0.757

The Random Forest classifier further improves performance with an accuracy of 78.9%. Its F1 Score of 77.9% and Jaccard Score of 75.7% indicate strong predictive power and better performance in balancing precision and recall.

- **Logistic Regression**
   - Accuracy Score: 0.792
   - F1 Score: 0.786
   - Jaccard Score: 0.757

Logistic Regression emerges as the best-performing model with the highest accuracy of 79.2%. Its F1 Score of 78.6% and Jaccard Score of 75.7% highlight its effectiveness in predicting customer churn. The model demonstrates a high level of reliability and balance between precision and recall.

### Best Model
Among the models evaluated, **Logistic Regression** achieved the best overall performance, with the highest accuracy score of 79.2%. Its F1 Score and Jaccard Score are also notably high, indicating its robustness in making accurate predictions and effectively identifying customer churn. While the Random Forest classifier also performed well, Logistic Regression's superior accuracy and balanced performance make it the most suitable model for predicting customer churn in this context.
 
### Key Insights
- **Logistic Regression:** Demonstrated the highest accuracy and well-balanced metrics, making it the most reliable model for predicting customer churn.
- **Random Forest Classifier:** Also showed strong performance with high accuracy and balanced precision and recall, making it a viable alternative.
- **K-Nearest Neighbors and Decision Tree Classifiers:** While they performed adequately, they were slightly less effective compared to Logistic Regression and Random Forest.
- The results underscore the importance of choosing the right model based on the specific metrics that matter most for the given problem, such as accuracy, precision, recall, and the ability to balance these metrics effectively.

## Usage
To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/HarshaEadara/Clustering-and-Churn-predition-of-Telco-Customer-Churn-Data-with-Machine-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Clustering-and-Churn-predition-of-Telco-Customer-Churn-Data-with-Machine-Learning
   ```
3. Install Dependencies:
Make sure you have Python installed. Then install the required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the Notebook:
Open the Jupyter Notebook and execute the cells
   ```bash
   jupyter notebook Clustering_and_Churn_predition_of_Telco_Customer_Churn_Data_with_Machine_Learning.ipynb
   ```
5. Ensure the dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` is available in the project directory.
6. Run the cells sequentially to execute the analysis.

## Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork this repository, make changes, and submit a pull request. Please ensure your code adheres to the project structure and is well-documented.


