import pandas as pd
from sklearn.model_selection import train_test_split
from CarResaleValue.DataTransformation.Transformation import DataAlteration

class SplitDepFeat:


    """

    Class Name : SplitDepFeat
    Description: This Class is used to split the dataset into training and testing sets.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.d = DataAlteration()
        self.d.DropColumn()
        self.d.DropRows()
        self.d.HandleCategoricalValues()


    def split(self):

        """

        Method_Name : split
        Description : Splitting the dataset into dependent and independent features.
        Output      : DataFrame
        On Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """
        try:
            self.x = self.d.data2.drop('Selling_Price', axis=1)
            self.y = self.d.data2['Selling_Price']
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y,test_size=0.30,random_state=100)
            return self.x_train,self.x_test,self.y_train,self.y_test
        except Exception as e:
            return e



