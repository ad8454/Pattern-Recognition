import sys
from Preprocess import view
from Classifiers import *
import numpy
import time
import pickle
from TrainedWeights import TrainWeight




def main():
    if len(sys.argv) < 1:
        print(len(sys.argv))
        print(" ERROR: wrong argument \n EXPEXTED: view-class.py <name of the file> <limit> ")
        exit(0)
    elif len(sys.argv) == 1:
        fileNameTrain = 'F:/Dev/PycharmProjects/Py34/PatternRecognition/Project1/real-train.csv'
    else:
        fileNameTrain = sys.argv[1]
        if len(sys.argv) >2:
            num=sys.argv[2]

    aView = view()
    num = 1000
    functions = [aView.zonning, aView.XaxisProjection, aView.YaxisProjection, aView.DiagonalProjections]
    train, classToNumber = aView.start(fileNameTrain,functions,num)
    print("Feature shape=",train.shape)
    numpy.random.shuffle(train)

    print('done with features!!!')
    start = time.time()
    rf=random_forest_train(train[:,:-1],train[:,-1])
    end = time.time()
    print("Training time for random forest",end-start)
    start = time.time()
    kd=kd_tree_train(train[:,:-1])
    end = time.time()
    print("Training time for KDTree", end - start)
    pickle.dump(TrainWeight(rf,kd,train[:,-1], classToNumber), open("TrainWeightFile.p", "wb"))

if __name__ == '__main__':
    main()