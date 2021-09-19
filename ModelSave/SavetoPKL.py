import pickle
from CarResaleValue.ModelSelection.ModelStacking import stacking
from CarResaleValue.ModelBuilding.model import model
from CarResaleValue.HyperParameterTuning.ModelTuning import tuning

class SerializedFile:

    def __init__(self):
        self.loc = 'model.pkl'
        self.mode = 'wb'
        self.m = model()
        self.s = stacking()
        self.t = tuning()

    def save_model(self):

        self.s.model_stacking()
        self.m.data_access()
        pickle.dump(self.s.stc,open(self.loc,self.mode))
        print("Success")






