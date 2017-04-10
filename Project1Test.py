import sys
from Preprocess import view
from Classifiers import *
import numpy
import pickle
import time
import csv

def main():
    if len(sys.argv) < 1:
        print(len(sys.argv))
        print(" ERROR: wrong argument \n EXPEXTED: view-class.py <name of the file> <limit> ")
        exit(0)
    elif len(sys.argv) == 1:
        fileNameTest = 'F:/Dev/PycharmProjects/Py34/PatternRecognition/Project1/real-test.csv'
    else:
        fileNameTest = sys.argv[1]
        aView = view()
        if len(sys.argv) >2:
            num=sys.argv[2]

    data = pickle.load(open("TrainWeightFile.p", "rb"))
    functions=[aView.zonning,aView.XaxisProjection,aView.YaxisProjection]
    aView=view()
    functions = [aView.zonning, aView.XaxisProjection, aView.YaxisProjection,aView.DiagonalProjections]
    test, opFileList = aView.start(fileNameTest,functions,[aView.OnlineFeature], True)
    #numpy.random.shuffle(test)
    print("Feature shape=", test.shape)
    print('done with features!!!')


    results = random_forest_test(data.RandomForest,test[:,:-1],test[:,-1])
    print_to_file('rf-results.csv', opFileList, results)

    results = kd_tree_test(data.KDTree,test[:,:-1],test[:,-1],data.GroundTruth)        #kd_tree(feature)
    results = [data.GroundTruth[index] for index in results]
    print_to_file('kdtree-results.csv', opFileList, results)

def print_to_file(fileName, opFileList, results):
    output = zip(opFileList, results)
    with open(fileName, 'w') as new_file:
        for inkml, label in output:
            new_file.write(inkml + ',' + label + '\n')

    print('File written to disk: ' + fileName)

if __name__ == '__main__':
    main()