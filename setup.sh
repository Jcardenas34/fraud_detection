#! /bin/bash/

# Gather necessary data
# if folder does nnot exist

if [ ! -d "dataset" ]; then
    echo "downloading dataset from kaggle"
    mkdir dataset
    curl -L -o $PWD/bank-transaction-dataset-for-fraud-detection.zip\
    https://www.kaggle.com/api/v1/datasets/download/valakhorasani/bank-transaction-dataset-for-fraud-detection
    unzip bank-transaction-dataset-for-fraud-detection.zip -d dataset
fi