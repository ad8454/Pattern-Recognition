import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class Classifiers:
    def random_forest_train(self, train_features, ground_labels):
        rf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
        rf = rf.fit(train_features, ground_labels)
        return rf

    def random_forest_test(self, rf, test_features, test_labels):
        scores = cross_val_score(rf, test_features, test_labels)
        print(scores)

    def kd_tree_train(self, train_features):
        kd = sklearn.neighbors.KDTree(train_features)
        return kd

    def kd_tree_test(self, kd, test_features, test_labels, ground_labels):
        score = 0
        for feature in test_features:
            dist, ind = kd.query(feature.reshape(1, -1), k=1)
            index = ind[0][0]
            if ground_labels[index] == test_labels[index]:
                score += 1

        print('kdtree score correct: ', score)
        print('kdtree score incorrect: ', len(test_features) - score)