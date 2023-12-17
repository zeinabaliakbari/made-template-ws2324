from sqlalchemy import create_engine
import pandas as pd
from pandas.io import sql
import sqlite3
import requests
import os
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.stats import zscore, iqr
class Pipeline:
 
    def __init__(self, file1_path, file2_path, output_directory):
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.output_directory = output_directory
        # self.table_name = table_name
        self.df1 = None
        self.df2 = None
       
       
    def read_csv_files(self):
        self.df2 = pd.read_csv(self.file2_path)
        self.df1 = pd.read_csv(self.file1_path)
        
        self.df1 = self.df1.drop(columns=['id'])
        new_column_names = {
            'age': 'Age',
            'bp': 'Blood_Pressure',
            'sg': 'Specific_Gravity',
            'al': 'Albumin',
            'su': 'Sugar',
            'rbc': 'Red_Blood',
            'pc': 'Pus_Cell',
            'pcc': 'Pus_Cell_clumps',
            'ba': 'Bacteria',
            'bgr': 'Blood_Glucose_Random',
            'bu': 'Blood_Urea',
            'sc': 'Serum_Creatinine',
            'sod': 'Sodium',
            'pot': 'Potassium',
            'hemo': 'Hemoglobin',
            'pcv': 'Packed_Cell_Volume',
            'wc': 'White_Blood_Cell_Count',
            'rc': 'Red_Blood_Cell_Count',
            'htn': 'Hypertension',
            'dm': 'Diabetes_Mellitus',
            'cad': 'Coronary_Artery_Disease',
            'appet': 'Appetite',
            'pe': 'Pedal_Edema',
            'ane': 'Anemia',
            'Outcome': 'Outcome',                 
            }

        self.df1 = self.df1.rename(columns=new_column_names)
        

    def normal_to_binary(self):
        # Iterate over columns
        for column in self.df1.columns:
            unique_values = self.df1[column].dropna().unique()
            if set(unique_values) == {'normal', 'abnormal'}:
                # Replace "normal" with 0, "abnormal" with 1, and missing values with the majority value
                self.df1[column] = self.df1[column].map({'normal': 0, 'abnormal': 1})
                majority_value = self.df1[column].mode().iloc[0]
                self.df1[column] = self.df1[column].fillna(majority_value)   
                         
    def present_to_binary(self):
        # Iterate over columns
        for column in self.df1.columns:
            unique_values = self.df1[column].dropna().unique()
            if set(unique_values) == {'notpresent', 'present'}:
                # Replace "notpresent" with 0, "present" with 1, and missing values with the majority value
                self.df1[column] = self.df1[column].map({'notpresent': 0, 'present': 1})
                majority_value = self.df1[column].mode().iloc[0]
                self.df1[column] = self.df1[column].fillna(majority_value)                                 
                        
    def good_to_binary(self):
            # Iterate over columns
        for column in self.df1.columns:
            unique_values = self.df1[column].dropna().unique()
            if set(unique_values) == {'good', 'poor'}:
                # Replace "good" with 0, "poor" with 1, and missing values with the majority value
                self.df1[column] = self.df1[column].map({'good': 0, 'poor': 1})
                majority_value = self.df1[column].mode().iloc[0]
                self.df1[column] = self.df1[column].fillna(majority_value)
 
    def outcome_to_binary(self ):
        self.df1.rename(columns={'classification': 'Outcome'}, inplace=True)

       # Map values in the column to 0 and 1
        self.df1['Outcome'] = self.df1['Outcome'].map({'notckd': 0, 'ckd': 1})
        
 

    def yesno_to_binary(self):
        for column in self.df1.columns:
            # Replace "yes" with 1 and "no" with 0
            self.df1[column] = self.df1[column].replace({'yes': 1, 'no': 0})
 

    def missing_with_majority(self):
        for column in self.df1.columns:
            unique_values = self.df1[column].dropna().unique()
            if set(unique_values) == {0, 1}:
                # Replace missing values with the majority value
                majority_value = self.df1[column].mode().iloc[0]
                self.df1[column] = self.df1[column].fillna(majority_value)
   

    def missing_with_mean(self ):
    # Convert all numeric columns to real (float)
        self.df1 = self.df1.apply(pd.to_numeric, errors='coerce')

        # Replace missing values with the mean for each column
        for column in self.df1.columns:
            mean_value = self.df1[column].mean()
            self.df1[column] = self.df1[column].fillna(mean_value)   
 
    def roundnumbers(self ):
        # Iterate over columns
        for column in self.df1.columns:
            # Check if the column is numeric
            if pd.api.types.is_numeric_dtype(self.df1[column]):
                # Convert values based on the specified conditions
                self.df1[column] = self.df1[column].apply(lambda x: 0 if x < 0.5 else (1 if x < 1 else x))
 

    def zero_with_mean_onecolumn(self ):

            # Columns to handle with Z-score
        zscore_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI']
        
        # Columns to handle with IQR
        iqr_columns = ['Pregnancies', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
        
        # Set the Z-score threshold
        zscore_threshold = (3, 5)
        
        # Set the IQR threshold
        iqr_threshold = (1.5, 3)
        
        # Handle outliers with Z-score
        for column in zscore_columns:
            z_scores = zscore(self.df2[column])
            self.df2[column] = self.df2[column][(z_scores > -zscore_threshold[0]) & (z_scores < zscore_threshold[1])]
        
        # Handle outliers with IQR
        for column in iqr_columns:
            q1 = self.df2[column].quantile(0.25)
            q3 = self.df2[column].quantile(0.75)
            iqr_range = iqr(self.df2[column])
            
            lower_bound = q1 - iqr_threshold[0] * iqr_range
            upper_bound = q3 + iqr_threshold[1] * iqr_range
            
            self.df2[column] = self.df2[column][(self.df2[column] > lower_bound) & (self.df2[column] < upper_bound)]
        
        # Drop rows with NaN values after handling outliers
        columns_to_convert = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

        self.df2[columns_to_convert] = self.df2[columns_to_convert].replace(0, pd.NA)
        self.df2 = self.df2.dropna()
         

        # Example usage:
        # df = pd.read_csv('your_dataset.csv')

        
         
 
    def merge_dataframes(self):
        merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)
        return merged_df

    def preprocess_data(self):
        # Display basic information about the dataset
        print("Dataset Info:")
        print(self.df1.info())



        # Handling outliers using Z-score
        print("\nHandling Outliers:")
        z_scores = np.abs(stats.zscore(self.df1))
        threshold = 3
        data_no_outliers = self.df1[(z_scores < threshold).all(axis=1)]

        print("Number of rows before removing outliers:", len(self.df1))
        print("Number of rows after removing outliers:", len(data_no_outliers))
        self.df1=data_no_outliers



    def save_to_sqlite(self, table_name1, table_name2):
        output_database_path = os.path.join(self.output_directory, 'madedb.sqlite')
        engine = create_engine(f'sqlite:///{output_database_path}')
        self.df1.to_sql(table_name1, engine, index=False, if_exists='replace')
        self.df2.to_sql(table_name2, engine, index=False, if_exists='replace')
        engine.dispose()
        print(f"SQLite database saved to: {output_database_path}")
        

    def run_pipeline(self, table_name1, table_name2):

        
        self.read_csv_files()
        if self.df1 is not None:
            self.outcome_to_binary()
            self.normal_to_binary()
            self.present_to_binary()
            self.good_to_binary()
            self.yesno_to_binary()
            self.missing_with_majority()
            self.missing_with_mean()
            self.roundnumbers()
            
           #
            self.zero_with_mean_onecolumn()
            self.preprocess_data()
            self.df1.to_csv('data/KidneyDisease.csv', index=False)
            self.df2.to_csv('data/diabetes.csv', index=False)
            #merged_df = data_pipeline.merge_dataframes(df1, df2)
            self.save_to_sqlite( table_name1, table_name2)

def main():
    file1_path = 'https://raw.githubusercontent.com/aiplanethub/Datasets/master/Chronic%20Kidney%20Disease%20(CKD)%20Dataset/ChronicKidneyDisease.csv'
    file2_path = 'https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv'
    tablename1 = Path(file1_path).stem
    tablename2 = Path(file2_path).stem
    output_directory = 'C:/Users/z004j5vt/made-template-ws2324/data/'
    pipeline = Pipeline(file1_path, file2_path, output_directory)
    pipeline.run_pipeline(tablename1, tablename2)

if __name__ == "__main__":
    main()
