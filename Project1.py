import sys
from Preprocess import view
from Classifiers import *
import numpy
import time

def main():
    if len(sys.argv) < 1:
        print(len(sys.argv))
        print(" ERROR: wrong argument \n EXPEXTED: view-class.py <name of the file> <limit> ")
        exit(0)
    elif len(sys.argv) == 1:
        aView = view()
        fileNameTrain = 'F:/Dev/PycharmProjects/Py34/PatternRecognition/Project1/real-train.csv'
        fileNameTest = 'F:/Dev/PycharmProjects/Py34/PatternRecognition/Project1/real-test.csv'
        #train = aView.start(file, [aView.zonning,aView.XaxisProjection,aView.YaxisProjection])
    else:
        fileNameTrain = sys.argv[1]
        fileNameTest = sys.argv[2]
        aView = view()
        if len(sys.argv) >3:
            num=sys.argv[3]

    num = 1000
    functions=[aView.hog]#, aView.zonning,aView.XaxisProjection,aView.YaxisProjection]
    train=aView.start(fileNameTrain,functions,num)
    print("Feature shape=",train.shape)
    numpy.random.shuffle(train)

    aView=view()
    functions = [aView.zonning, aView.XaxisProjection, aView.YaxisProjection]
    test=aView.start(fileNameTest,functions,num)
    numpy.random.shuffle(test)
    print("Feature shape=", test.shape)

    print('done with features!!!')
    start = time.time()
    rf=random_forest_train(train[:,:-1],train[:,-1])
    end = time.time()
    print("Training time for random forest",end-start)
    random_forest_test(rf,test[:,:-1],test[:,-1])
    start = time.time()
    kd=kd_tree_train(train[:,:-1])
    end = time.time()
    print("Training time for KDTree", end - start)
    kd_tree_test(kd,test[:,:-1],test[:,-1],train[:,-1])        #kd_tree(feature)

if __name__ == '__main__':
    main()