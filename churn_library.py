# library doc string
"""
Library to predict customer churn
Author: Jack Huang
Date: 28-Jan-22
"""

# import libraries
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import constants
sns.set()

class Model:
    """A class to represent a model pipeline given a csv file
    """
    version: str
    _df_raw: pd.DataFrame
    _df_cleaned: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame
    rfc: RandomForestClassifier
    lrc: LogisticRegression

    def __init__(self, version: str):
        """Initiate model with version name

        Args:
            version (str): version name of model for model dump names
        """
        self.version = version

    def import_data(self, pth: str) -> None:
        '''
        returns dataframe for the csv found at pth

        input:
                pth: a path to the csv
        output:
                None. initiates class attribute df
        '''
        print(f"Importing data from {pth}...")
        self._df_raw = pd.read_csv(pth)

    def process_df(self) -> None:
        """Pipeline to clean df including:
        1) Create churn column
        2) Encode categorical columns with churn proportions
        """
        assert self._df_raw is not None, "No raw data loaded!"

        print("Cleaning raw data...")
        self._df_cleaned = self._df_raw.copy()
        self._create_target_col()
        self._encode_proportion()

    def _create_target_col(self) -> None:
        """Add Churn column to df
        """
        print("   Creating target feature Churn...")
        self._df_cleaned['Churn'] = self._df_cleaned['Attrition_Flag'].apply(
            lambda val: 0 if val == 'Existing Customer' else 1)

    def _encode_proportion(
            self,
            response: str = '_Churn') -> None:
        '''
        transforms self.df. helper function to turn each categorical column into a new column with
        propotion of churn for each category

        input:
            df: pandas dataframe
            response: string of response name [naming variables or index y column]
        output:
            None. Transforms self.df.
        '''
        print("   Encoding categorical features...")
        for col in constants.cat_columns:
            group = self._df_cleaned.groupby(col).mean()['Churn']
            self._df_cleaned[f'{col}{response}'] = self._df_cleaned[col].map(lambda x: group.loc[x])

    def perform_eda(
            self,
            output_folder: str = constants.path_eda):
        """Perform EDA on columns and output to folders

        Args:
            histogram_cols (str): columns names for histogram plot.
            value_count_cols (str): columns names for value count plot.
            displot_cols (str): columns names for displot.
            output_folder (str): path for model output.
        """
        assert self._df_cleaned is not None, "No cleaned data loaded!"

        print(f"Performing eda... saving to {output_folder}...")
        # histograms
        for cols in constants.eda_histogram_columns:
            plt.figure(figsize=(20, 10))
            self._df_cleaned[cols].hist()
            plt.savefig(f"{output_folder}{cols}_histogram.png")

        # value counts
        for cols in constants.eda_valcount_columns:
            plt.figure(figsize=(20, 10))
            self._df_cleaned[cols].value_counts('normalize').plot(kind='bar')
            plt.savefig(f"{output_folder}{cols}_value_count.png")

        # distplot
        for cols in constants.eda_distplot_columns:
            plt.figure(figsize=(20, 10))
            sns.displot(self._df_cleaned[cols])
            plt.savefig(f"{output_folder}{cols}_distplot.png")

        # heatmap
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self._df_cleaned.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig(f"{output_folder}Heatmap.png")

    def perform_feature_engineering(
            self,
            test_size: float = constants.test_size
    ) -> None:
        '''
        input:
                df: pandas dataframe
                cols: base columns for features

        output:
                None. intiates class attributes
                _X_train: X training data
                _X_test: X testing data
                _y_train: y training data
                _y_test: y testing data
        '''
        assert self._df_cleaned is not None, "No cleaned data loaded!"
        print("Performing feature engineering...")
        # duplicated X from selected columns
        self.X = self._df_cleaned[constants.feature_cols]
        self.y = self._df_cleaned['Churn']

        # train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=constants.random_state)

    def train_models(self) -> None:
        """fit models with loaded train/test splits, and store models

        Args:
            param_grid (dict): Params for GridCV to search best estimator for RFC.
			Defaults to constants.param_grid.
        """
        assert self.X_train is not None, "No X_train data!"
        assert self.y_train is not None, "No y_train data!"
        print("Training Random Forest Classifier...")
        # CV fit rfc
        rfc = RandomForestClassifier(random_state=constants.random_state)
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=constants.param_grid, verbose=1)
        cv_rfc.fit(self.X_train, self.y_train)
        self.rfc = cv_rfc.best_estimator_

        print("Training Logistic Regression...")
        # fit logistic regression
        self.lrc = LogisticRegression(max_iter=1000)
        self.lrc.fit(self.X_train, self.y_train)

        # store models
        rfcpath = f"{constants.path_model}rfc_model_v{self.version}.pkl"
        print(f"Saving RFC model to {rfcpath}...")
        joblib.dump(self.rfc, rfcpath)

        lrcpath = f"{constants.path_model}lrc_model_v{self.version}.pkl"
        print(f"Saving LRC model to {lrcpath}...")
        joblib.dump(self.lrc, lrcpath)

    def predict(self, data: pd.DataFrame) -> dict:
        """predict target given cleaned dataset. Outputs results from random forest ('rfc') and
		log reggression ('lrc')

        Args:
            data (pd.DataFrame): X features matrix

        Returns:
            dict: model outputs, with keys being 'rfc' for random forest and
			'lrc' for logistic regression
        """

        assert self.rfc is not None, "No Random Forest Classifier trained!"
        assert self.lrc is not None, "No Logistic Regression trained!"
        y_preds_rf = self.rfc.predict(data)
        y_preds_lr = self.lrc.predict(data)
        return {'rfc': y_preds_rf, 'lrc': y_preds_lr}

    @staticmethod
    def output_report(
            y: pd.Series,
            y_preds: pd.Series,
            title: str,
            filename: str,
            output_folder: str = constants.path_results) -> None:
        """Output image of classification report based on y and y_predctions

        Args:
            y (pd.Series): target
            y_preds (pd.Series): y predictions
            title (str): title for the output
            filename (str): name of file
            outpath (str, optional): folder to save output
        """
        plt.figure('figure', figsize=(5, 2.5))
        plt.text(
            0.01, 1, str(title), {
                'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y, y_preds)),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.axis('off')
        plt.savefig(f"{output_folder}{filename}")
        plt.close()

    def classification_report_image(self) -> None:
        '''
        produces classification report for training and testing results and stores report as image
        in images folder

        output:
                None
        '''
        train_preds = self.predict(self.X_train)
        test_preds = self.predict(self.X_test)
        reports = {
            'Random Forest': {
                'Train': train_preds['rfc'],
                'Test': test_preds['rfc']},
            'Logistic Regression': {
                'Train': train_preds['lrc'],
                'Test': test_preds['lrc']}}
        y = {'Train': self.y_train, 'Test': self.y_test}
        for model_name, preds in reports.items():
            for train_test, pred in preds.items():
                self.output_report(
                    y[train_test],
                    pred,
                    model_name +
                    " " +
                    train_test,
                    model_name.lower().replace(
                        ' ',
                        '_') +
                    "_" +
                    train_test +
                    "_v" +
                    self.version +
                    ".png")

    def feature_importance_plot(
            self, output_path: str = constants.path_results) -> None:
        '''
        creates and stores the feature importances in pth
        input:
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = self.rfc.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.X.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self.X.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self.X.shape[1]), names, rotation=90)
        plt.savefig(output_path + 'feature_importance.png')
        plt.close()
    
    def roc_plot(self, output_path: str = constants.path_results) -> None:
        """plot roc curve of the Random Forest and Logit models stored based on test results

        Args:
            output_path (str, optional): path to save the plots
        """
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(self.rfc, self.X_test, self.y_test, ax=ax, alpha=0.8)
        lrc_plot = plot_roc_curve(self.lrc, self.X_test, self.y_test)
        rfc_disp.plot()
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(output_path + 'ROC.png')
        plt.close()

    def load_saved_model(self, path: str, classifier: str) -> None:
        """Load pre-trained models into the model class

        Args:
            path (str): path for saved model
            classifier (str): 'lrc' for logistic regression, 'rfc' for random forest
        """
        if classifier == 'rfc':
            self.rfc = joblib.load(path)
        elif classifier == 'lrc':
            self.rfc = joblib.load(path)
        else:
            print("classifier must be either rfc for Random Forest or lrc for Logit!")

    def get_df_raw(self) -> pd.DataFrame:
        """getter for df_raw"""
        return self._df_raw

    def get_df_cleaned(self) -> pd.DataFrame:
        """getter for df_cleaned"""
        return self._df_cleaned


if __name__ == '__main__':
    model = Model('1.0')
    model.import_data(constants.path_rawdata)
    model.process_df()
    model.perform_eda()
    model.perform_feature_engineering()
    model.train_models()
    model.classification_report_image()
    model.feature_importance_plot()
    model.roc_plot()