# Paths
path_rawdata = r"./data/bank_data.csv"
path_eda = r"./images/eda/"
path_results = r"./images/results/"
path_model = r"./models/"
path_logs = r"|./logs/"

# Features
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'                
]

quant_columns = [
    'Customer_Age',
    'Dependent_count', 
    'Months_on_book',
    'Total_Relationship_Count', 
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 
    'Credit_Limit', 
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 
    'Total_Amt_Chng_Q4_Q1', 
    'Total_Trans_Amt',
    'Total_Trans_Ct', 
    'Total_Ct_Chng_Q4_Q1', 
    'Avg_Utilization_Ratio'
]

feature_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                'Income_Category_Churn', 'Card_Category_Churn'
                ]

# EDA
eda_histogram_columns = ['Churn','Customer_Age']
eda_valcount_columns = ['Marital_Status']
eda_distplot_columns = ['Total_Trans_Ct']

# Model parameters
test_size = 0.3  # train test split
random_state = 42  # random state for random forest

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}