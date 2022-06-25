from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

class Resampler:
    def __init__(self, class_distribution, over, under):
        self.distrib = class_distribution
        majority_samples = max(self.distrib.values())
        
        undersampling_strategy = "not minority"
        self.oversampling_strategy = {0: majority_samples, 1: int(lie_down_samples * 1.1), 2: int(fall_samples*1.2)}
        
        if over is "smote":
            self.oversampler = SMOTE()
        