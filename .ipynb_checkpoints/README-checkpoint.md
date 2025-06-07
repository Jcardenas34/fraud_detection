# Fraud Detection for Credit/Debit Card transactions

This repo presents the work done to perform a statistical analysis on open source Credit/Debit card transaction data for fraud detection.

Original dataset for the project was obtained here
https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection


Two methods were exercised in this git repo, a purely statistical approach consisting of mostly Z-Score analysis, as well as a more sophisticated approach to fraud detection using a Deep learning algorithm known as an Auto-encoder network.

## Using this repo
This analysis was primarily conducted using jupyter notebooks, you will first have to download the git repo by running
and can be run running
```
git clone git@github.com:Jcardenas34/fraud_detection.git
cd fraud_detection
source setup.sh
```
which will download the dataset for your use. From there run the [statistical_anamolies.ipynb](./statistical_anamolies.ipynb), and [fraud_detection_autoencoder.ipynb](./fraud_detection_autoencoder.ipynb) notebooks. They are heavily commented for your use.

# Z-Score Analysis
In statistics, there is a concept known as a Z-Score that represents how many standard deviations a value is from the mean of its distribution. If a sampling distribution can be converted into a gaussian, we can determine the likelihood of observing any given event by using a Z-Score. A Z-Score of 0 given to a value means that it lies perfectly at the mean of the distribution, and so it is highly likely that it would be observed. But values that are less probable have higher Z-Scores, and so Z-Scores can be used to flag data as anomalous. All events with a Z-Score within +-2 represent about 95% of the values in that distribution, and so data points with a Z score larger than this can be said to have a probability of being observed of about 5%. For events with Z-Scores within +-3, the probability of observance falls to about .03%. With this knowledge, we can state that values with a Z-Score larger than 2, occur infrequently, but events with a Z-score values of 3 or greater are extremely unlikely, unlikely enough to be suspicious. In this study, I flagged events as potentially fraudulent if any of their numerical features had a Z-Score > 3. Lets see what insights this choice revealed.



## Interpretations of the Z-Score analysis.
Using this purely statistical approach, I was able to determine that there were 140 out of 2512 data entries had characteristics of fraud, or 5.57% of the sample.
After determining the cases of fraud using Z-Scores, I was able to determine that the city in the US with the most fraudulent cases was FortWorth, TX with a total of 7 cases in a dataset os 2412 throughout the United States. 

![!\[Image 1\](plots/.png)](plots/instances_of_fraud_by_city.png)

Additionally, by using this Z-Score methodology, I was able to determine that there were 2 variables of interest in determining fraud, that showed a high number of events with Z-Scores larger than 3 in their respective distributions. They are "TransactionAmounts" and "LoginAttempts". Thinking about this logically, it makes sense that a very large withdrawl could indicate fraud, as well as a high number of login attempts could indicate difficulty inputting a password, and so could also indicate a fraudulent transaction. Below are 2D scatter plots of these two variables along with others where we can gain insight on the kinds of transactions that were flagged as fraud. A complete set of plots used for this portion of the analysis can be found in the plots folder [here](plots/).

| Transaction Amount Focus          | Login Attempts Focus           |
|--------------------|--------------------|
| ![!\[Image 1\](plots/.png)](plots/TransactionAmount_vs_CustomerAge.png) | ![!\[Image 2\](plots/.png)](plots/LoginAttempts_vs_CustomerAge.png)|

From these plots we can see a clear threshold at where the Z-Score of 3 is defined for the Transaction amount, as well as the number of Login attempts. We find that cases of fraud are spread uniformly across age groups, and tend to be in higher amounts. 



# Multi-Variate analysis using an Autoencoder
Autoencoders provide an effective method for detecting anomalies in data by learning to reconstruct input data as accurately as possible. In the case where we have a dataset with many ordinary events, where only a small number are "anamolous", using an autoencoder makes sense. When the network is trained using many examples of ordinary data, it can learn to reconstruct the ordinary instances well, and anamolous data poorly.

By using the Mean Squared Error (MSE) as a loss function, the network will learn to reconstruct events based on the vast majority of input examples which are presumed to be non-fraudulent. Events with anamolous characteristics, will be reconstructed poorly, and so create an indicator by which we can detect fraud. By specifying a threshold for the MSE, we can create a boundary by which events above the threshold can be flagged as fraudulent.



![!\[Image 3\](plots/.png)](plots/mse_zscore_AE.png)

Here you can see that I have arbitrarily chosen the value of 2 to be the threshold of anamoly detection for the MSE, as we can see that a vast majority of the reconstructed feature vectors lie below 2, and so MSE values above two can be considered anamolous. 

## Interpretations of the Multi-Variate analysis using an Autoencoder

A complete set of plots used for this portion of the analysis can be found in the plots folder [here](autoencoder_plots/).
| Transaction Amount Focus          | Login Attempts Focus           |
|--------------------|--------------------|
| ![!\[Image 4\](plots/.png)](autoencoder_plots/LoginAttempts_vs_TransactionWeekNumber.png) | ![!\[Image 5\](plots/.png)](autoencoder_plots/LoginAttempts_vs_AccountBalance.png)|
