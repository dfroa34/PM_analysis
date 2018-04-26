import configparser
import json
import csv
import pandas as pd

import sys
import warnings
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error,mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVR, SVR
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
import graphviz
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import numpy as np
import seaborn as sns


if not sys.warnoptions:
    warnings.simplefilter("ignore")


# -----------------------------------------------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------------------------------------------


#Pipeline for Regression
r_pipelines = {
        'Linear': Pipeline([('DT',linear_model.LinearRegression())]),
        'DT':  Pipeline([('DT', DecisionTreeRegressor())]),
        'SVM': Pipeline([('DT',  SVR(kernel='linear'))])
    }

#Pipeline for Classification
c_pipelines = {
        # Attention - name pipeline according to classifier name
        'NB': Pipeline([('NB', MultinomialNB())]),
        'LR':  Pipeline([('LR', LogisticRegression(multi_class='multinomial', solver='lbfgs'))]),
        'SVM': Pipeline([('SVM',  SVC())]),
        'DT': Pipeline([('DT',  tree.DecisionTreeClassifier())])
    }

#Hyperparameter optimization for the classification algorithms in c_pipeline
nb_hyperparameters = {
    'NB__alpha': [0.2, 0.5, 0.8, 1.0]
}

lr_hyperparameters = {
    'LR__C': [1]  # [1, 0.2, 0.5, 0.8]
}

svm_hyperparameters = {
    'SVM__C': [1],  # [0.1, 0.5, 1, 5],
    'SVM__kernel': ['linear']  # ['linear', 'rbf']
}

dt_hyperparameters = {
    'DT__criterion': ['gini', 'entropy'],
    'DT__max_depth': np.arange(1, 12, 2),  # [3,5,7],#[5, 8, 11, 14],
    'DT__min_samples_leaf': np.arange(10, 100, 10),  # [10,20,50,100],#[1, 3, 5]
}

hyperparameters = {
        'NB': nb_hyperparameters,
        'LR': lr_hyperparameters,
        'SVM': svm_hyperparameters,
        'DT': dt_hyperparameters
}

#Config file path
config_path = 'config_file.txt'

# Results folder path
results_path = './results/'


# -----------------------------------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------------------------------

def processJSON():
    '''
    Process the JSON file and store the results in the results path given in the config file.
    '''

    #Load input file
    config = configparser.ConfigParser()
    config.read(config_path)
    log_inf = config['app-info']
    input_file = log_inf['inputSurroundingsFile']

    result = open(results_path + 'surroundings.csv', 'w', newline='')
    output_file = csv.writer(result)

    with open(input_file) as data_file:
        data = json.load(data_file)

    #Print header of CSV file
    header = ['store_code'] + [surrounding for surrounding in data[0]['surroundings']]
    output_file.writerow(header)

    #Print the rest of the file
    for store in data:
        result = []
        result.append(store['store_code'])
        for surrounding in store['surroundings']:
            result.append(1) if  len(store['surroundings'][surrounding]) >0 else result.append(0)
        output_file.writerow(result)


def processSales(aggregation_type):
    '''
    Process the raw sales file. The function removes duplicates, transform columns to the right data type (numeric values to integers, date values to date_type) and aggregates\
    the data according to the given parameter. (M - Month, A- year, Q - quarter)
     It generates a CSV file with the sales
    with the transformed dataset.
    '''

    # Load input file
    config = configparser.ConfigParser()
    config.read(config_path)
    log_inf = config['app-info']
    input_sales_file = log_inf['inputSalesFile']

    df = (pd.read_csv(input_sales_file, header=None))

    #Remove duplicates
    df = df.drop_duplicates(subset=0)

    #Transpose dataset: it is necessary to perform data aggregation over the date_time values.
    data = (df.iloc[:, 1:]).T
    header = ((df).iloc[:, :1]).T
    headers = header.values[0]
    headers = [str(i) for i in headers]

    data.columns = headers
    data = data.rename(index=str, columns={"store_code": "date_time"})

    #Data type transformation
    data['date_time'] = pd.to_datetime(data['date_time'])

    # Transform each field except the date_time to a numeric value
    for header in headers[1:]:
        data[[header]] = data[[header]].apply(pd.to_numeric)

    # Data aggregation
    results = pd.DataFrame()
    for header in headers[1:]:
        results[header] = data.set_index('date_time').resample(aggregation_type)[header].sum()

    results.to_csv(results_path + 'sales_per_' + aggregation_type + '.csv', sep=',')



def join_files():
    '''
    Join the two transformed files from the previous functions.
    :return: final_dataset.csv containing the final input data for the classification/regression models.
    '''
    sales = pd.read_csv(results_path + 'sales_per_year.csv', delimiter=',')
    surroundings = pd.read_csv(results_path + 'surroundings.csv', delimiter=',')
    surroundings['store_code'] = surroundings['store_code'].apply(pd.to_numeric)

    salesT = (sales.iloc[:, 1:]).T
    header = ((sales).iloc[:, :1]).T
    headers = header.values[0]
    headers = [str(i) for i in headers]

    salesT.columns = headers
    salesT['store_code'] = salesT.index
    salesT['store_code'] = salesT['store_code'].apply(pd.to_numeric)

    #Join results
    print('-----------------------------------------------------------------------------------------')
    result = pd.merge(surroundings, salesT, on='store_code')
    result.to_csv('./results/final_dataset.csv')
    print(result)


def run_benchmark(task_type):
    '''
    Runs several classification/regression algorithms on the final dataset.
    :param task_type: type of task for choosing the benchmark pipeline (regression, classification)
    '''
    print('Running classifiers...')

    #According to the task_type the input dataset for the model differs. Additionally, the algorithms used to perfom the classification/regression as well.
    pipelines = {}
    if task_type == 'R':
        dataset = 'final_dataset.csv'
        pipelines = r_pipelines
    else:
        dataset = 'final_dataset_class.csv'
        pipelines = c_pipelines

    data = pd.read_csv(results_path + dataset)

    #1. Prepare data
    #Select the rows you want to skip in your input data for the model
    cols = [col for col in data.columns if col not in ['Total', 'store_code',  '12/31/2016', '12/31/2015', '12/31/2017']]
    x_raw = data[cols].apply(pd.to_numeric)

    #Select the column you want to analyze as a label. In this case, we will analyze at the total
    y_raw = data['Total'].apply(pd.to_numeric)


    # 2. Split data into training and test
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.20,
                                                        random_state=42, shuffle=True)
    #3. Run pipelines
    for name, pipeline in pipelines.items():
        # Fit model on X_train and y_train
        print('-----------------------------------------------')
        print(name)

        if task_type == 'C':
            # Create cross-validation object
            model = (GridSearchCV(pipeline, hyperparameters[name], cv=10))

        model = pipeline.fit(x_train, y_train)
        model = model.best_estimator_

        predicted = model.predict(x_test)
        if task_type  == 'R':
            print(str(name) + ',' + str(mean_squared_error(y_test, predicted))) #Print the metric for measuring the regression task. It could be MSE, MAE... etc.
        else:
            print('Accuracy: ' + str(accuracy_score(y_test, predicted)))


def visualize_decision_tree():
    '''
    Run the decision tree classifier and plots the results using graphviz. The graphviz tree is stored in the results folder.
    '''

    print('Running decision tree visualization...')

    # Load data
    pipelines = c_pipelines
    dataset = 'final_dataset_class.csv'

    data = pd.read_csv(results_path + dataset)

    # 1. Prepare data
    # Select the rows you want to skip in your input data for the model
    cols = [col for col in data.columns if col not in ['Total', 'store_code', '12/31/2016', '12/31/2015', '12/31/2017']]
    x_raw = data[cols].apply(pd.to_numeric)

    # Select the column you want to analyze as a label. In this case, we will analyze at the total
    y_raw = data['Total'].apply(pd.to_numeric)

    # 2. Split data into training and test
    x_train, x_test, y_train, y_test = train_test_split(x_raw, y_raw, test_size=0.20,
                                                        random_state=42, shuffle=True)

    # Create Decision tree classifier
    model = tree.DecisionTreeClassifier(max_features=10, max_depth=4) #Parameter tunning based on the benchmark


    # Fit model on X_train and y_train
    model = model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    print(model.feature_importances_)
    print('Accuracy: ' + str(accuracy_score(y_test, predicted)))

    #Export render to graphviz. In case graphviz is not installed in your computer, you can visualize the tree by copying the content of the file in the following
    #website: https://dreampuf.github.io/GraphvizOnline/
    dot_data = tree.export_graphviz(model, out_file=None,
                                    feature_names=x_raw.columns,
                                    class_names=['0', '1'],
                                    filled=True, rounded=True,
                                    special_characters=True)

    graph = graphviz.Source(dot_data)
    graph.render(results_path + 'graphviz_result')



def run_association_rules():
    '''
    Finds association rules by using the apriori algorithm for finding frequent itemsets on the final dataset. The final rules are stored in the results file.
    Take into account that the min_support parameter on the frequent_itemset generates all the possible combinations of attributes that meet this parameters. A low support means that
    the algorithm will take a lot of time to process all possible options.
    '''
    print('Running association rules...')
    x_raw = pd.read_csv(results_path + 'final_dataset_ar.csv')

    frequent_itemsets = apriori(x_raw, min_support=0.7, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    rules.to_csv(results_path + 'associationRules.csv')


def run_correlation():
    '''
    Calculates and visualize the correlations between all the final variables.
    The results are stored in the results folder
    '''

    x_raw = pd.read_csv(results_path + 'final_dataset_ar.csv')
    corr = x_raw.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom  colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    corr.to_csv(results_path + 'correlations.csv')



#All the functions are independent from each other (if the previous data is already generated). That is why the results are stored in CSV files instead of passing
#data frames to the following function. The idea is to show the progress and discuss the results at each step.
#Please comment/uncomment any of the functions if you just want to see any of the individual results.

#processJSON()
#processSales()
#join_files()
#run_benchmark('C')
visualize_decision_tree()
#run_association_rules()
#run_correlation()
