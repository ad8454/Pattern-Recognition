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
        aView.start('F:/Dev/PycharmProjects/Py34/PatternRecognition/Project1/filtered_x_2.csv')
    else:
        fileName = sys.argv[1]
        aView = view()
        num=1000
        if len(sys.argv) >2:
            num=sys.argv[2]
        functions=[]
        feature=aView.start(fileName,[aView.zonning,aView.XaxisProjection,aView.YaxisProjection],num)
        print("Feature shape=",feature.shape)
        numpy.random.shuffle(feature)
        size = len(feature)*70//100
        train = feature[:size]
        print('train: ', numpy.unique(train[:,-1]))
        test = feature[size:]
        print('test: ', numpy.unique(test[:,-1]))

        rf=random_forest_train(train[:,:-1],train[:,-1])
        random_forest_test(rf,test[:,:-1],test[:,-1])
        kd=kd_tree_train(train[:,:-1])
        kd_tree_test(kd,test[:,:-1],test[:,-1],train[:,-1])        #kd_tree(feature)

if __name__ == '__main__':
    main()