from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from CarResaleValue.DataDistribution.DataSplit import SplitDepFeat

class model:

    """

    Class_Name : access_train_test_data
    Description: This class is used to access training and testing data and then using this data to train the Machine Learning Model
    Written by : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def data_access(self):

        """

        Method_Name : data_access
        Description : This method is used to access the training and testing data
        Output      : Dataframe
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.x_train,self.x_test,self.y_train,self.y_test = SplitDepFeat().split()
        except Exception as e:
            return e

    def rfc_model(self):

        """

        Method_Name : rfc_model
        Description : This method is used to train the data using Random Forest Regressor
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.rfr = RandomForestRegressor()
            self.rfr.fit(self.x_train,self.y_train)
        except Exception as e:
            return e

    def gbr_model(self):

        """

        Method_Name : gbr_model
        Description : This method is used to train the data using Gradient Boosting Regressor.
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.gbr = GradientBoostingRegressor()
            self.gbr.fit(self.x_train,self.y_train)
        except Exception as e:
            return e

    def xgbr_model(self):

        """

        Method_Name : xgbr_model
        Description : This method is used to train the data using Xgboost Regressor
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.xgbr = XGBRegressor()
            self.xgbr.fit(self.x_train,self.y_train)
        except Exception as e:
            return e





