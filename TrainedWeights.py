

class TrainWeight:
    __slots__ = 'RandomForest', 'KDTree', 'GroundTruth'

    def __init__(self, RF, KD, GT):
        self.RandomForest = RF
        self.KDTree = KD
        self.GroundTruth = GT