from sklearn import metrics
from CarResaleValue.ModelBuilding.model import model

class ModelEvaluation:

    """

    Class_Name : ModelEvaluation
    Description: This class is used to evaluate the model based on training and testing data.
    Written by : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.m = model()
        self.m.data_access()

    def rfr_eval(self):

        """

        Method_Name : rfr_eval
        Description : This method is used to evaluate the performance Random Forest Regressor Model based on training and testing data.
        Output      : Float Values
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.m.rfc_model()
            self.y_pred = self.m.rfr.predict(self.m.x_test)
            print("Training Score :- ",self.m.rfr.score(self.m.x_train,self.m.y_train))
            print("Training Score :- ", self.m.rfr.score(self.m.x_test, self.m.y_test))
            print('MAE:', round(metrics.mean_absolute_error(self.m.y_test, self.y_pred), 4))
            print('MSE:', round(metrics.mean_squared_error(self.m.y_test, self.y_pred), 4))
        except Exception as e:
            return e


    def gbr_eval(self):

        """

        Method_Name : gbr_eval
        Description : This method is used to evaluate the performance Gradient Boosting Regressor Model based on training and testing data.
        Output      : Float Values
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.m.gbr_model()
            self.y_pred = self.m.gbr.predict(self.m.x_test)
            print("Training Score :- ",self.m.gbr.score(self.m.x_train,self.m.y_train))
            print("Training Score :- ", self.m.gbr.score(self.m.x_test, self.m.y_test))
            print('MAE:', round(metrics.mean_absolute_error(self.m.y_test, self.y_pred), 4))
            print('MSE:', round(metrics.mean_squared_error(self.m.y_test, self.y_pred), 4))
        except Exception as e:
            return e


    def xgbr_eval(self):

        """

        Method_Name : xgbr_eval
        Description : This method is used to evaluate the performance XGBoost Regressor Model based on training and testing data.
        Output      : Float Values
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        self.m.xgbr_model()
        self.y_pred = self.m.xgbr.predict(self.m.x_test)
        print("Training Score :- ", self.m.xgbr.score(self.m.x_train, self.m.y_train))
        print("Training Score :- ", self.m.xgbr.score(self.m.x_test, self.m.y_test))
        print('MAE:', round(metrics.mean_absolute_error(self.m.y_test, self.y_pred), 4))
        print('MSE:', round(metrics.mean_squared_error(self.m.y_test, self.y_pred), 4))

