# Machine-Learnin--Churn-prediction

# Credit Card Customer Churn Analysis

This repository contains the code and documentation for my machine learning project on credit card churn prediction, hosted on Kaggle. The dataset used can be found [here on Kaggle](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers).

## Project Overview

The objective of this project is to predict customer churn using various machine learning classification models and perform customer segmentation using K-means clustering. This project was completed solo over a period of 4 days, focusing on rapid development and deployment of predictive models.

## Models Implemented

The following machine learning models were implemented for the classification task:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Decision Tree
- Random Forest

For clustering, K-means was utilized, with an analysis to find the optimal number of clusters using the elbow method.

## Key Features Analyzed

Significant features were identified and analyzed to understand their impact on customer churn:
- **Total_Revolving_Bal**: The total revolving balance on the credit card.
- **Total_Trans_Ct**: The total count of transactions.
- **Total_Trans_Amt**: The total amount of transactions.
- **Total_Ct_Chng_Q4_Q1**: Change in transaction count from Q4 to Q1.
- **Total_Amt_Chng_Q4_Q1**: Change in transaction amount from Q4 to Q1.

## Visualizations

Visualizations were created using the Decision Tree model due to its comparative effectiveness. The visualizations focus on:
- Distribution and impact of key features on churn.
- Customer segmentation results.
- Trends in transaction activities showing customers with a downward trend.

## Business Interventions

Based on the analysis, several business interventions were proposed to mitigate churn and enhance customer engagement, focusing on the most impactful features.

## Clustering Analysis

The clustering analysis includes:
- **Elbow Method**: Used to determine the optimal number of clusters. This involved plotting the sum of squared distances from each point to their respective cluster center and finding the 'elbow point' where the rate of decrease sharply shifts.
- **Silhouette Score Evaluation**: After determining the optimal number of clusters using the elbow method, silhouette scores were calculated to evaluate the effectiveness of the clustering, ensuring the clusters are well-separated and cohesive.

## Classification Model Evaluation

Each classification model was rigorously evaluated using metrics such as accuracy, precision, recall, F1-score, and the ROC-AUC score. The evaluations helped in selecting the best performing model for this specific dataset.


## Contributing

Feel free to fork this project and submit a pull request if you have suggestions for improvements or new features.

## Author

**Mahsa Nazarian**

For any further inquiries or collaboration requests, feel free to contact me 

## Usage

To replicate the findings or extend the analysis, clone this repository and install the required packages listed in `requirements.txt`.

```bash
git clone https://github.com/mahsanazar/Machine-Learning-Churn-prediction
cd credit-card-churn-analysis
pip install -r requirements.txt















