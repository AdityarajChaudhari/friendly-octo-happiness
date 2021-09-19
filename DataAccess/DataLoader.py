import pandas as pd

class GetData:

    """
    Class_Name : GetData
    Description: This class is used to access the dataset.
    Written By : Adityaraj Hemant Chaudhari
    Versions   : 0.1
    Revision   : None

    """

    def __init__(self):
        self.data_src = r"C:\Users\LEGION\ML Projects\vehiclesales\CarResaleValue\Data\car data.csv"
    def acquire_data(self):

        """

        Method_Name : acquire_data
        Description : This method is used to acquire data from the respective data path.
        Output      : DataFrame
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revision    : None

        """

        try:
            self.data = pd.read_csv(self.data_src)
            return self.data
        except Exception as e:
            raise e

g = GetData()
g.acquire_data()
