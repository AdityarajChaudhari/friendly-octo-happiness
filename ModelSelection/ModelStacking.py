from sklearn.ensemble import StackingRegressor
from CarResaleValue.HyperParameterTuning.ModelTuning import tuning
from xgboost import XGBRegressor
from CarResaleValue.ModelBuilding.model import model
from sklearn.ensemble import BaggingRegressor
from sklearn import metrics

class stacking:

    def __init__(self):
        self.m = model()
        self.t = tuning()
        self.m.data_access()

    def model_stacking(self):
        self.m.xgbr_model()
        self.t.gbr_tuning()
        self.t.rfr_tuning()
        #self.bgr = BaggingRegressor()
        self.stc = StackingRegressor(estimators=[('randomcv_rfr_best',self.t.randomcv_rfr_best),('randomcv_gbr_best',self.t.gbr_best)], final_estimator = self.m.xgbr)
        self.stc.fit(self.m.x_train , self.m.y_train)
        print("Training Score :- ",self.stc.score(self.m.x_train,self.m.y_train))
        print("Testing Score :- ",self.stc.score(self.m.x_test,self.m.y_test))

