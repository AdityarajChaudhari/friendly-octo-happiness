import numpy as np
import pandas as pd
from CarResaleValue.DataAccess.DataLoader import GetData

class DataAlteration:

    """

    Class_Name : DataAlteration
    Description: This class is used to clean the dataset and remove unnecessary columns and rows which will not contribute in building generalized model.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = GetData().acquire_data()

    def DropColumn(self):

        """

        Method Name : DropColumn
        Description : This method is used to drop irrelevant column
        Output      : DataFrame
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """
        try:
            self.data1 = self.data.drop(['Car_Name','Year'],axis=1)
            return self.data1
        except Exception as e:
            return e

    def DropRows(self):

        """

        Method_Name : DropRows
        Description : This method is used to drop irrelevant rows from the dataset.(Here rows containing CNG and 3 data is removed as they are rare values and won't contribute much to model.
        Output      : DataFrame
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            cng_index = self.data1[self.data1['Fuel_Type']=='CNG'].index
            #print(cng_index)
            owner3index = self.data1[self.data1['Owner']==3].index
            #print(owner3index)
            self.data2 = self.data1.drop(self.data.index[[18,35,85]])
            return self.data2
        except Exception as e:
            return e

    def HandleCategoricalValues(self):

        """

        Method Name : HandleCategoricalValues
        Description : The categorical values from some columns like Owner, Transmission,etc are transformed into numerical values.
        Output      : Pandas DataFrame
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """
        try:
            self.x = pd.get_dummies(self.data2[['Fuel_Type','Seller_Type','Transmission']],drop_first=True)
            #print(self.x)
            self.data2 = pd.concat([self.data2, self.x], axis=1)
            self.data2.drop(['Fuel_Type', 'Seller_Type', 'Transmission'], axis=1, inplace=True)
            print(self.data2)
        except Exception as e:
            raise e


