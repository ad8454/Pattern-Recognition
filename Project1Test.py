import sys
from Preprocess import view
from Classifiers import *
import numpy
import pickle
import time

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

    aView=view()
    functions = [aView.zonning, aView.XaxisProjection, aView.YaxisProjection, aView.DiagonalProjections]
    data=pickle.load(open("TrainWeightFile.p", "rb"))
    test=aView.start(fileNameTest,functions,num, data.classToNumber)
    numpy.random.shuffle(test)
    print("Feature shape=", test.shape)
    print('done with features!!!')
    random_forest_test(data.RF,test[:,:-1],test[:,-1])
    kd_tree_test(data.KD,test[:,:-1],test[:,-1],data.GroundTruth)        #kd_tree(feature)

if __name__ == '__main__':
    main()