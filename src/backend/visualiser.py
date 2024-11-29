import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualiser:
    def create_correlation_heatmap(data: pd.DataFrame, save_file_path: str):
        """ Creates a correlation heatmap showing the correlation between different features
        and the target feature and saves it as a png at the specified save_file_path
        """
        corr = data.corr()
        sns.heatmap(corr)
        plt.savefig(save_file_path)

    def create_class_distribution_bar_graph(data: pd.DataFrame, save_file_path: str):
        """ Creates a bar graph showing the how data is distributed
        across the classes
        """
        counts = data['loan_status'].value_counts()
        counts = counts.replace(to_replace=0, value='Not in default')
        counts = counts.rename({ 0: 'Not in default', 1: 'In default' })
        sns.barplot(counts)
        plt.savefig(save_file_path)
    
    def create_loan_interest_histogram(data: pd.DataFrame, save_file_path: str):
        """ Creates a histogram showing how interests rates are
        distributed
        """
        plt.hist(data)
        plt.savefig(save_file_path)
    
    def create_piechart_showing_home_ownership(data: pd.DataFrame, save_file_path: str):
        """ Creates a piechart showing different home ownership types
        and their distributions
        """
        counts = data['person_home_ownership'].value_counts()
        plt.pie(counts, labels=counts.axes[0])
        plt.savefig(save_file_path)