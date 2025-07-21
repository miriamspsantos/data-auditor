import pandas as pd
from profiler import DataProfiler
from itertools import combinations


class FairnessProfiler:
    def __init__(self, df, sensitive_features:list[str], target:str=None):
        self.df = df
        #TODO: maybe check if these features exist in df
        self.sensitive_features = sensitive_features
        self.target = target
        self.profiler = DataProfiler(df=df, target=target)
        self.sensitive_groups = FairnessProfiler.intersectional_fairness_groups(sensitive_features)
     
    def group_proportions_table(self, group:str, plot_tab=False) -> pd.DataFrame: 
        """Creates a table of group distribution.
        Group can be a combination of sensitive features, given by
        self.sensitive_groups.
        """
        if group in self.sensitive_groups.keys():
            sensitive_features = self.sensitive_groups[group]
            rows_index = sensitive_features[0] if len(sensitive_features) > 1 else None
            cols = sensitive_features[1:] if rows_index else sensitive_features
            
        pivot = pd.pivot_table(
            self.df,
            index=rows_index,
            columns=cols,
            aggfunc="size",
            fill_value=0
        )

        pivot_percent = round(pivot / pivot.values.sum() * 100, 2)

        if plot_tab:
                self.profiler.plot_pivot_table(
                    index=rows_index,
                    columns=cols,
                    aggfunc="size", 
                    title="Proportions by Group",
                    plot_percent=True, 
                    figsize=(pivot_percent.shape[1] * 2, pivot_percent.shape[0] * 2))

        return pivot_percent

    def target_rate_by_group_table(self, group:str, target:str=None, plot_tab=False) -> pd.DataFrame:
        """Creates a table of target distribution by group.
        Group can be a combination of sensitive features, given by
        self.sensitive_groups.
        """
        #TODO: add protections for unknown group or target
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

            if plot_tab:
                self.profiler.plot_pivot_table(
                    index=target,
                    columns=self.sensitive_groups[group],
                    aggfunc="size", 
                    title="Proportions by Group",
                    plot_percent=True, 
                    figsize=(pivot_rate.shape[1] * 2, pivot_rate.shape[0] * 2))

            return pivot_rate

    def missing_data_disparity_imbalance(self):
        """Calculates the disparity (imbalance ratio) by feature
        for each group or group aggregation. 
        Returns a dictionary for all groups and all features.
        """
        pass
    
    def plot_table(self):
        """Represents pivot_tables in a visual format."""
        # TODO: Call from profiler with specific settings, adjust figsize dynamically
        # TODO: embed as a flag in main method or keep separate?
        pass

    def plot_sensitive_groups(self):
        pass

    def get_sensitive_groups(self):
        return list(self.sensitive_groups.keys())
    
    def get_sensitive_features(self):
        return self.sensitive_features


    @staticmethod
    def intersectional_fairness_groups(sensitive_features:list[str]):
        """Determines interseccional sensitive groups.
        For each sensitive group, returns the features that
        create it.
        """
        res = []
        for i in range(1, len(sensitive_features) + 1):
            c = list(combinations(sensitive_features, i))
            res.extend(c)
        group_lst = ["_".join(pairs) for pairs in res]

        intersectional_groups = {}
        for g in group_lst:
            intersectional_groups[g] = g.split("_")
        
        return intersectional_groups