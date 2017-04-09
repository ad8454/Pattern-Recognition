"""
Program to read in txt files with ground truth values and
randomly partitioning the data into training and testing sets.

Author: Ajinkya Dhaigude (ad8454@rit.edu)
"""

import sys
import csv
from random import shuffle


def partition(ground_real_txt, ground_junk_txt, t):
    """
    Splits the input data into training and testing sets

    :param ground_real_txt: input file name for ground truth values
    :param ground_junk_txt: input file for junk values
    :param t: percentage of all data to be split into testing set
    :return: None
    """

    partition_helper(ground_real_txt, 'real', t)
    partition_helper(ground_junk_txt, 'junk', t)


def partition_helper(file_txt, type, t):
    """
    Helper function to split the data

    :param file_txt: input file
    :param type: string indicating 'real' or 'junk' data
    :param t: percentage of all data to be split into testing set
    :return: None
    """

    with open(file_txt, 'r') as file:
        ground_truths = list(csv.reader(file))

    # randomize the data order
    shuffle(ground_truths)


    # create a dict for all unique symbols -- symbols: [inkml names]
    all_symbols = {}
    for truth in ground_truths:
        true_class = truth[1].strip()
        if true_class not in all_symbols.keys():
            all_symbols[true_class] = []
        all_symbols[true_class].append(truth[0])

    # write test data set to disk
    with open(type + '-test.csv', 'w') as new_file:
        count=0
        count_train=0
        for key in all_symbols.keys():
            count+=1
            # compute testing set size as an integer
            testing_size = len(all_symbols[key]) * t // 100
            for ele in all_symbols[key][:testing_size]:
                new_file.write(str(ele) + ',' + key + '\n')
                count_train+=1

    # write training data set to disk
    with open(type + '-train.csv', 'w') as new_file:
        count_test = 0
        for key in all_symbols.keys():
            # compute training set size as an integer
            testing_size = len(all_symbols[key]) * t // 100
            for ele in all_symbols[key][testing_size:]:
                new_file.write(str(ele) + ',' + key + '\n')
                count_test+=1
    print("Total",count_train+count_test)
    print("Training size",count_test)
    print("Testing size", count_train)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Command line arguments do not match.\nRunning program with default parameters...')
        partition('task2-trainSymb2014/trainingSymbols/iso_GT.txt', 'task2-trainSymb2014/trainingJunk/junk_GT.txt', 30)
    else:
        partition(str(sys.argv[1]), str(sys.argv[2]), int(sys.argv[3]))

    print('Done. Files written to disk:\nreal-train.csv\nreal-test.csv\njunk-train.csv\njunk-test.csv')
