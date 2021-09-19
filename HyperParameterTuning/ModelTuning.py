from sklearn.model_selection import RandomizedSearchCV
from CarResaleValue.ModelBuilding.model import model
import numpy as np

class tuning:
    """

    Class_Name : ModelEvaluation
    Description: This class is used to Tune the model by setting some parameter Values.
    Written by : Adityaraj Hemant Chaudhari
    Version    : 0.1
    Revisions  : None

    """

    def __init__(self):
        self.m = model()
        self.m.data_access()

    def rfr_tuning(self):

        """

        Method_Name : rfr_tuning
        Description : This method is used to tune the Random Forest Regressor Model by setting some parameter values.
        Output      : Float Values
        On_Failure  : Raise Exception

        Written By  : Adityaraj Hemant Chaudhari
        Version     : 0.1
        Revisions   : None

        """

        try:
            self.m.rfc_model()
            self.params = {
                'criterion': ['mse', 'friedman_mse'],
                'max_features': ['sqrt', 'auto', 'log2', None],
                'max_depth': [i for i in range(1, 50, 1)],
                'n_estimators': [i for i in range(100, 3000, 30)],
                'min_samples_split': [i for i in range(2, 50, 1)],
                'min_samples_leaf': [i for i in range(1, 50, 1)]
            }
            self.randomcv_rfr = RandomizedSearchCV(self.m.rfr, param_distributions=self.params, n_jobs=-1, cv=5, n_iter=100, verbose=True)
            self.randomcv_rfr.fit(self.m.x_train,self.m.y_train)
            self.randomcv_rfr_best = self.randomcv_rfr.best_estimator_
            self.y_pred1 = self.randomcv_rfr_best.predict(self.m.x_test)
            print("Training_Score :- ",self.randomcv_rfr_best.score(self.m.x_train,self.m.y_train))
            print("Testing_Score :- ",self.randomcv_rfr_best.score(self.m.x_test,self.m.y_test))
        except Exception as e:
            return e


    def gbr_tuning(self):

        """

               Method_Name : gbr_tuning
               Description : This method is used to tune the Gradient Boosting Regressor Model by setting some parameter values.
               Output      : Float Values
               On_Failure  : Raise Exception

               Written By  : Adityaraj Hemant Chaudhari
               Version     : 0.1
               Revisions   : None

               """
        try:
            self.m.gbr_model()
            self.param_grid = {
                'learning_rate': [float(x) for x in np.linspace(0.1, 1, 10)],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                'max_features': ['auto', 'sqrt'],
                'min_samples_leaf': [i for i in range(1, 50, 1)],
                'min_samples_split': [i for i in range(2, 50, 1)],
                'n_estimators': [int(i) for i in np.linspace(100, 3000, 30)],
                'alpha': [a for a in np.linspace(0.1, 0.99, 20)],
                'loss': ['ls', 'lad', 'huber', 'quantile']
            }
            self.randomcv_gbr = RandomizedSearchCV(estimator=self.m.gbr, param_distributions=self.param_grid, n_iter=100, cv=5, n_jobs=-1,verbose=True)
            self.randomcv_gbr.fit(self.m.x_train, self.m.y_train)
            self.gbr_best = self.randomcv_gbr.best_estimator_
            self.y_pred1 = self.gbr_best.predict(self.m.x_test)
            print("Training_Score :- ", self.gbr_best.score(self.m.x_train, self.m.y_train))
            print("Testing_Score :- ", self.gbr_best.score(self.m.x_test, self.m.y_test))
        except Exception as e:
            return e

