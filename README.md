# Fraud Detection for Credit/Debit Card transactions
An repo to compare the performance of various unsupervised learning models (Auto encoder, Variational Auto encoder, Isolation Forest) to flag anamolous credit card transactions using the (valakhorasani fraud dataset)[https://www.kaggle.com/datasets/valakhorasani/bank-transaction-dataset-for-fraud-detection]. 

# Motivation
Detecting anamolies in high stakes transactions such as in credit card transations are critical in building trust between clients and providers. Fincncial firms often use a plethora of deep learning models to detect and sometimes anticipate fraudulent activity. My aim is to explore the the performance of traditional ML models, deep learning models and LLM driven anamoly detection to understand weather there is a meaningful difference in their ability to flag fraudulent activity, and maximize fraudulent activity rejection. 

# Autoencoder
 A purely statistical approach consisting of Z-Score analysis, as well as a more sophisticated approach to fraud detection using a Deep learning algorithm known as an Auto-encoder network.

## Quickstart
This analysis was primarily conducted using Jupyter Notebooks, you will first have to download the git repo by running
and can be run running
```
git clone git@github.com:Jcardenas34/fraud_detection.git
cd fraud_detection
source setup.sh
```
which will download the dataset for your use. From there run the [statistical_anamolies.ipynb](./statistical_anamolies.ipynb), and [fraud_detection_autoencoder.ipynb](./fraud_detection_autoencoder.ipynb) notebooks. They are heavily commented for your use.

## Docker
This repo comes with a working docker contner so that a webserver that hosts the Autoencoder can take in data from a provided data set. Run the containers like such (Designed for a Jetson Nano Jetpack 4.5)
```
sudo docker build -t fraud-detection-nano .
```
```
sudo docker run --runtime nvidia --security-opt seccomp=unconfined -p 8000:8000 fraud-detection-nano
```

# Fraud Rate Analysis
## Autoencoders: Z-Score Analysis
  Let's see what insights this choice revealed.



## Interpretations of the Z-Score analysis.
Using this purely statistical approach, I was able to determine that there were 140 out of 2512 data entries had characteristics of fraud or 5.57% of the sample.
After determining the cases of fraud using Z-Scores, I was able to determine that the city in the US with the most fraudulent cases was FortWorth, TX with a total of 7 cases in a dataset of 2412 throughout the United States. 

![!\[Image 1\](plots/.png)](plots/instances_of_fraud_by_city.png)

Additionally, by using this Z-Score methodology, I was able to determine that there were 2 variables of interest in determining fraud, that showed a high number of events with Z-Scores larger than 3 in their respective distributions. They are "TransactionAmounts" and "LoginAttempts". Thinking about this logically, it makes sense that a very large withdrawal could indicate fraud, as well as a high number of login attempts, which could indicate difficulty inputting a password, and so could also indicate a fraudulent transaction. Below are 2D scatter plots of these two variables along with others where we can gain insight on the kinds of transactions that were flagged as fraud. A complete set of plots used for this portion of the analysis can be found in the plots folder [here](plots/).

| Transaction Amount Focus          | Login Attempts Focus           |
|--------------------|--------------------|
| ![!\[Image 1\](plots/.png)](plots/TransactionAmount_vs_CustomerAge.png) | ![!\[Image 2\](plots/.png)](plots/LoginAttempts_vs_CustomerAge.png)|

From these plots, we can see a clear threshold where the Z-Score of 3 is defined for the Transaction amount, as well as the number of Login attempts. We find that cases of fraud are spread uniformly across age groups, and tend to be in higher amounts. 



# Multi-variate analysis using an Autoencoder
Autoencoders provide an effective method for detecting anomalies in data by learning to reconstruct input data as accurately as possible. In the case where we have a dataset with many ordinary events, where only a small number are "anomalous", using an autoencoder makes sense. When the network is trained using many examples of ordinary data, it can learn to reconstruct the ordinary instances well, and anomalous data poorly.

By using the Mean Squared Error (MSE) as a loss function, the network will learn to reconstruct events based on the vast majority of input examples which are presumed to be non-fraudulent. Events with anomalous characteristics will be reconstructed poorly, and so create an indicator by which we can detect fraud. By specifying a threshold for the MSE, we can create a boundary by which events above the threshold can be flagged as fraudulent. In this study, I flagged events as potentially fraudulent if any of their numerical features had a Z-Score > 3.



![!\[Image 3\](plots/.png)](plots/mse_zscore_AE.png)

Here you can see that I have chosen the value of 2 to be the threshold of anomaly detection for the MSE. Here I transformed the MSE values into a z-score distribution to provide a statistically motivated way to select the anomaly threshold. We can see that a vast majority of the reconstructed feature vectors lie below 2, where 2 represents the boundary under which 95% of the data lie, and so MSE values above 2 can be considered anomalous. 

## Interpretations of the Multi-Variate analysis using an Autoencoder

A complete set of plots used for this portion of the analysis can be found in the plots folder [here](autoencoder_plots/).
| Transaction Amount Focus          | Login Attempts Focus           |
|--------------------|--------------------|
| ![!\[Image 4\](plots/.png)](autoencoder_plots/LoginAttempts_vs_TransactionWeekNumber.png) | ![!\[Image 5\](plots/.png)](autoencoder_plots/LoginAttempts_vs_AccountBalance.png)|


# Multi-variate analysis using a Variational Autoencoder

Variational auto encoders provide an alternative way to perform anomaly detection by which the input that is trying to be reconstructed by the network is mapped onto a probability distribution as a latent space, instead of a fixed latent representation.

Work in progress..

