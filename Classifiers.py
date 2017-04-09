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
    #scores = cross_val_score(rf, test_features, test_labels)

    #scores=rf.score(test_features,test_labels)
    #print('rf result', scores)

    #scores=rf.score(test_features,test_labels)
    score=0
    prediction, bias, contributions = rf.predict(test_features)
    for index in range(len(test_labels)):
        classPredict = numpy.argmax(prediction[index])
        if classPredict == test_labels[index]:
            score+=1
    print('Random Forest score correct: ', score)
    print('Random Forest incorrect: ', len(test_labels) - score)
    print('Accuracy:',(score/len(test_labels)*100))

def kd_tree_train(train_features):
    kd = sklearn.neighbors.KDTree(train_features)
    return kd

def kd_tree_test(kd, test_features, test_labels, ground_labels):
    score = 0
    label_idx=0
    for feature in test_features:
        dist, ind = kd.query(feature.reshape(1, -1), k=1)
        index = ind[0][0]
        if ground_labels[index] == test_labels[label_idx]:
            score += 1
        label_idx+=1

    print('kdtree score correct: ', score)
    print('kdtree score incorrect: ', len(test_features) - score)
    print('kdtree accuracy: ',(score/len(test_features)*100))