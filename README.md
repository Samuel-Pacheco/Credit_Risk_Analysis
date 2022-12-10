# Credit_Risk_Analysis

## Overview
The purpose of this analysis is to understand how to use Machine Learning statistical algorithms to make predictions based on data patterns provided. In this challenge, we focus on Supervised Learning using a free dataset from LendingClub, a P2P lending service company to evaluate and predict credit risk. 

To complete this analysis, we use different Machine Learning techniques to train and evaluate the data with unbalanced classes. The dataset from the LendingClub has an unbalanced classification problem due to the number of good loans outweighing the amount of risky loans. In order balance out the classifications to allow for more meaningful predictions and improve the accuracy score, we needed to employ various Machine Learning algorithms to resample the data. These algorithms include RandomOverSampler, SMOTE, ClusterCentroids, SMOTEENN, BalancedRandomForestClassifier, and EasyEnsembleClassifier.
## Results
As said, We have used machine learning to sample the dataset using  Python libraries Scikit-learn and imbalanced -learn to evaluate the results and compare.
The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk"

![image](https://user-images.githubusercontent.com/53358476/206849310-2af5add3-da2d-45c4-97c3-67a67d67b04c.png)

Oversampling RandomOverSampler Model randomly selects from the minority class and adds it to the training set until both classifications are equal.

![image](https://user-images.githubusercontent.com/53358476/206849383-6d50f39d-b55b-446d-88df-7f0072b726e9.png)

The balanced accuracy score improved slightly to 68%.

![image](https://user-images.githubusercontent.com/53358476/206849464-9a085ee6-2198-4576-8fd3-6313e090f6e0.png)

The "High Risk" precision rate was only 0.88 with the recall at 0.37 giving this model an F1 score of 52. "Low Risk" had a precision rate of 100% and recall at 100%.

![image](https://user-images.githubusercontent.com/53358476/206849576-5ee37e9d-7590-459d-9a01-bae7b244e280.png)

## Summary
All the models used to perform the credit risk analysis show weak precision in determining if a credit risk is high.\
The Ensemble models brought a lot more improvment specially on the sensitivity of the high risk credits.\
The EasyEnsembleClassifier model shows a recall of 88% so it detects almost all high risk credit. On another hand, with a low precision, a lot of low risk credits are still falsely detected as high risk which would penalize the bank's credit strategy and infer on its revenue by missing those business opportunities.\
For those reasons I would not recommend the bank to use any of these models to predict credit risk.

