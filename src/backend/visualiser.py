import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Visualiser:
    def create_correlation_heatmap(data: pd.DataFrame, save_file_path: str):
        """ Creates a correlation heatmap showing the correlation between different features
        and the target feature and saves it as a png at the specified save_file_path
        """
        plt.figure()
        corr = data.corr()
        sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm', annot=True)
        plt.title('Correlation Heatmap')
        plt.savefig(save_file_path)

    def create_class_distribution_bar_graph(data: pd.DataFrame, save_file_path: str):
        """ Creates a bar graph showing the how data is distributed
        across the classes
        """
        data = data['loan_status'].rename({ 0: 'Not in default', 1: 'In default' })
        plt.figure()
        sns.barplot(data['loan_status'])
        plt.savefig(save_file_path)
    
    def create_loan_interest_histogram(data: pd.DataFrame, save_file_path: str):
        """ Creates a histogram showing how interests rates are
        distributed
        """
        plt.figure()
        plt.hist(data)
        plt.savefig(save_file_path)
    
    def create_piechart_showing_home_ownership(data: pd.DataFrame, save_file_path: str):
        """ Creates a piechart showing different home ownership types
        and their distributions
        """
        plt.figure()
        counts = data['person_home_ownership'].value_counts()
        plt.pie(counts, labels=counts.axes[0])
        plt.savefig(save_file_path)