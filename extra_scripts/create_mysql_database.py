#! /usr/env/ python

import pandas as pd
import mysql.connector

'''
Description:
------------
A script that converts a CSV file into a MySQL database table.
It is used to convert the credit card fraud detection dataset from Kaggle into a MySQL database table.

First the CSV is loaded into a Pandas dataframe, the dataframe specs are obtained, and then the 
data rows are looped over to write the data to the MySQL database table.

#! Note, prerequisite:
You must first create a database in MySQL before running this script, go to a clean bash shell.
mysql -u root -p
CREATE DATABASE fraud_detection;

'''

# Location of the CSV file
csv_file = "/Users/chiral/git_projects/fraud_detection/dataset/bank_transactions_data_2.csv"

# MySQL database credentials, needed to access mysql
db_config = {
    "host": "localhost",        # Replace with your host
    "user": "root",             # Replace with your MySQL username
    "password": "K!tsun3=F0x?!1",# Replace with your MySQL password
    # "password": "youd-love-to-know",# Replace with your MySQL password
    "database": "fraud_detection" # Replace with your database name
}

# Table name
table_name = "customer_transactions"

# Read the CSV file into a Pandas DataFrame
# Begin debug
df = pd.read_csv(csv_file)
# print(df.head())
print(df.columns)
# Dataset column has incorrect formatting (IP Address) and needs to be changed to IPAddress
df.rename(columns={'IP Address': 'IPAddress'}, inplace=True)


# Connect to the MySQL database
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# Create the table based on the DataFrame schema
columns = ", ".join([f"{col} TEXT" for col in df.columns])
# Specifying the table name and the content of the columns, they are all text columns for now
create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns});"
cursor.execute(create_table_query)

# Loop over the rows in the pandas dataframe to insert data into the table
for i, row in df.iterrows():
    placeholders = ", ".join(["%s"] * len(row))
    insert_query = f"INSERT INTO {table_name} VALUES ({placeholders})"
    # does the actual writing of the data
    cursor.execute(insert_query, tuple(row))

# Commit changes and close the connection
conn.commit()
cursor.close()
conn.close()

print(f"Table '{table_name}' has been created and populated in the database '{db_config['database']}'.")
