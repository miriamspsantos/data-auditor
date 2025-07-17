import pandas as pd
from profiler import DataProfiler
from itertools import combinations


class FairnessProfiler:
    def __init__(self, df, sensitive_features:list[str], target:str=None):
        self.df = df
        self.sensitive_features = sensitive_features
        self.target = target
        self.profiler = DataProfiler(df=df, target=target)
        self.sensitive_groups = FairnessProfiler.intersectional_fairness_groups(sensitive_features)
     
    def group_proportions_table(self):
        """Creates a table of group distribution.
        Sensitive features are shown individually and in combination.
        """
        pass

    def target_rate_by_group_table(self):
        """Creates a table of target distribution by sensitive features.
        Sensitive features are shown individually and in combination.
        """
        pass

    def missing_data_disparity_imbalance(self):
        """Calculates the disparity (imbalance ratio) by feature
        for each group or group aggregation. 
        Returns a dictionary for all groups and all features.
        """
        pass
    
    def plot_feature_by_group(self):
        """Plots a feature by group."""
        pass

    def get_sensitive_groups(self):
        return self.sensitive_groups
    
    def get_sensitive_features(self):
        return self.sensitive_features


    @staticmethod
    def intersectional_fairness_groups(sensitive_features:list[str]):
        res = []
        for i in range(1, len(sensitive_features) + 1):
            c = list(combinations(sensitive_features, i))
            res.extend(c)
        group_lst = ["_".join(pairs) for pairs in res]
        return group_lst