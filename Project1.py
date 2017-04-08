import sys
from Preprocess import view
from Classifiers import *

def main():
    if len(sys.argv) < 1:
        print(len(sys.argv))
        print(" ERROR: wrong argument \n EXPEXTED: view-class.py <name of the file> <limit> ")
        exit(0)
    elif len(sys.argv) == 1:
        aView = view()
        file = 'F:/Dev/PycharmProjects/Py34/PatternRecognition/Project1/filtered_x_2.csv'
        aView.start(file, [aView.hog])
    else:
        fileName = sys.argv[1]
        aView = view()
        num=10
        if len(sys.argv) >2:
            num=sys.argv[2]
        functions=[]
        feature=aView.start(fileName,[aView.Histogram],num)

        random_forest_train(feature[:,:-1],feature[:,-1])
        #kd_tree(feature)

if __name__ == '__main__':
    main()