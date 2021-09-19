import numpy as np
import pandas as pd
from CarResaleValue.DataAccess.DataLoader import GetData

class OverView:

    """

    Class_Name : Overview
    Description: This class is used for understanding data i.e. no.of rows , no.of columns ,etc.
    Written By : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.data = GetData().acquire_data()

    def get_shape(self):

        """

        Method Name : get_shape
        Description : This method is used to get the shape of the DataFrame
        Output      : Numpy Array (2-D Array)
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.shape = self.data.shape
            return self.shape
        except Exception as e:
            return e

    def get_info(self):

        """

        Method_Name : get_info
        Description : This method is used to find the info i.e data type of columns,check null values , values present in column,etc.
        Output      : Pandas DataFrame
        On_Failure  : Raise Exception

        Written by  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """
        try:
            self.info = self.data.info()
            return self.info
        except Exception as e:
            return e

o = OverView()
o.get_shape()