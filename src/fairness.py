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
     
    def group_proportions_table(self) -> pd.DataFrame: 
        """Creates a table of group distribution.
        Sensitive features are shown individually and in combination.
        """
        

    def target_rate_by_group_table(self, group:str, target:str=None) -> pd.DataFrame:
        """Creates a table of target distribution by group.
        Group can be a combination of sensitive features, given by
        self.sensitive_groups.
        """
        
        target = self.target if target is None else target 
            
        if group in self.sensitive_groups.keys() and target:
            pivot = pd.pivot_table(
                self.df, 
                index=target,
                columns=self.sensitive_groups[group],
                aggfunc="size",
                fill_value=0
            )

            pivot_rate = round(pivot / pivot.values.sum() * 100, 2)
            return pivot_rate

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
        return list(self.sensitive_groups.keys())
    
    def get_sensitive_features(self):
        return self.sensitive_features


    @staticmethod
    def intersectional_fairness_groups(sensitive_features:list[str]):
        res = []
        for i in range(1, len(sensitive_features) + 1):
            c = list(combinations(sensitive_features, i))
            res.extend(c)
        group_lst = ["_".join(pairs) for pairs in res]

        intersectional_groups = {}
        for g in group_lst:
            intersectional_groups[g] = g.split("_")
        
        return intersectional_groups