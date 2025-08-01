import os
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer   #used to create pipeline for transformation
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function do data transformation
        """
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_columns=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(      #this we are creating pipeline for numerical col to fill and normalize
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),     #this is for filling missing value  
                    ("scaler",StandardScaler())                 #this is for standardizing
                ]
            )

            logging.info("Nuemrical columns standard scaling completed")

            cat_pipeline=Pipeline(   #this pipeline for dealing with categorical things
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False)) 
                ]
            )

            logging.info("Categorical columns encoding completed")
            
            #for combining both numerical and categorical pipeline we will use columnTransformer
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        """
        This function is applying preprocessing to train and test datset
        """
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()   #obj of above function of preprocessing

            target_column_name="math_score"
            numerical_columns=["writing_score","reading_score"]

            #spliting into dependent and independent column for train_df
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            #splitting into dependent and independent for test_df
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing for training and test datframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)   #using object to transform dataframe
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            #try to convert input into np.c_  this c_ stack array side wise side
            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing objects.")

            save_object(    #saving the pickle file #this save_object function is defined in utils
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)


        
