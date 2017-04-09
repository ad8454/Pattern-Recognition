

class TrainWeight:
    __slots__ = 'RandomForest', 'KDTree', 'GroundTruth'

    def __init__(self, RF, KD, GT, classToNumber):
        self.RandomForest = RF
        self.KDTree = KD
        self.GroundTruth = GT
        self.classToNumber = classToNumber