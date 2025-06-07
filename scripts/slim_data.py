#! /usr/bin/python


import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def set_x_y_limits(plot_array, x, y):
    for plot in plot_array:
        plot.set_xlim(0, x)
        plot.set_ylim(1, y)
        plot.semilogy()


def main(args):
    '''
    Description::
    ------------
    A script that plots important features of the fraud detection dataset.

    '''
    
    df = pd.read_csv(args.data)
    print(df.columns)
    # for k in df.columns:
        # print(k)

    print(df.head())
    occupations = df["CustomerOccupation"].unique()
    print(occupations)

    student_habits = df[df["CustomerOccupation"] == "Student"]
    doctor_habits = df[df["CustomerOccupation"] == "Doctor"]
    engineer_habits = df[df["CustomerOccupation"] == "Engineer"]
    retiree_habits = df[df["CustomerOccupation"] == "Retired"]

    # plot_grid = plt.figure()
    # plot_grid.suptitle("Customer Habits")
    fig, axs = plt.subplots(2, 2)

    student_habits.hist(column='TransactionAmount', ax=axs[0, 0])
    doctor_habits.hist(column='TransactionAmount', ax=axs[0, 1])
    engineer_habits.hist(column='TransactionAmount', ax=axs[1, 0])
    retiree_habits.hist(column='TransactionAmount', ax=axs[1, 1])
    axs[0,0].set_title("Student")
    axs[0,1].set_title("Doctor")
    axs[1,0].set_title("Engineer")
    axs[1,1].set_title("Retired")

    plot_array = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]   

    set_x_y_limits(plot_array, 2000, 1000)
    plt.suptitle("Customer Habits")
    plt.show()
    # saving the plot
    plt.savefig("customer_habits.png")
    plt.close()
    # print(df.head())

    # print(doctor_transactions.head())

    # plt.hist(doctor_transactions["TransactionType"], bins=2)

    # debit = doctor_transactions[doctor_transactions["TransactionType"] == "Debit"]
    # credit = doctor_transactions[doctor_transactions["TransactionType"] == "Credit"]

    # df.hist(column='TransactionAmount')
    # df.hist(column='TransactionType')
    # plt.show()
    # print(debit.head())

    

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("data", help="Data file to be read")
    args = argparse.parse_args()

    main(args)