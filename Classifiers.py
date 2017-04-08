import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy

#class Classifiers:
def random_forest_train(train_features, ground_labels):
    rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    rf = rf.fit(train_features, ground_labels)
    return rf

def random_forest_test(rf, test_features, test_labels):
    scores = cross_val_score(rf, test_features, test_labels)
    print(scores)

def kd_tree_train(train_features):
    kd = sklearn.neighbors.KDTree(train_features)
    return kd

def kd_tree_test(kd, test_features, test_labels, ground_labels):
    score = 0
    #test_list = list(test_features)
    label_idx=0
    for feature in test_features:
        #print(test_list.index(feature)[0])
        #label_idx = numpy.where(test_features==feature)[0][0]
        dist, ind = kd.query(feature.reshape(1, -1), k=1)
        index = ind[0][0]
        #print(ground_labels[index] , test_labels[label_idx])
        if ground_labels[index] == test_labels[label_idx]:
            score += 1
        label_idx+=1

    print('kdtree score correct: ', score)
    print('kdtree score incorrect: ', len(test_features) - score)
    print('Accuracy:',(score/len(test_features)*100))