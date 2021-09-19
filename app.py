import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle
import sklearn
from flask_cors import CORS, cross_origin

app = Flask(__name__)
car_model = pickle.load(open('CarResaleValue/ModelSave/model.pkl', 'rb'))
#print(car_model)

@cross_origin()
@app.route('/',methods=['GET'])
def home():
    return render_template('./home.html')
print("Inside HomePage")

@cross_origin()
@app.route('/my-app',methods=['GET'])
def index_page():
    print("Inside My Application")
    return render_template('./index.html')

@cross_origin()
@app.route('/my-app/cars75',methods=['GET'])
def car_app():
    print("Inside Index Page")
    return render_template('./carresalevalueapp.html')

@cross_origin()
@app.route('/my-app/cars75/predict',methods=['GET','POST'])
def predictcarapp():
    if request.method == 'POST':
        Present_Price = float(request.form['Showroom Price'])
        Kms_Driven = int(request.form["No.of KM's Driven"])
        Kms_Driven2 = np.log(Kms_Driven)
        second_hand = request.form['Is_the_car_second_hand']
        if second_hand == 'No':
            second_hand = 0
        else:
            second_hand = 1
        seller_type = request.form['Type of Seller']
        if seller_type == 'Individual':
            seller_type = 0
        else:
            seller_type = 1
        fuel_type = request.form['Type of Fuel']
        if fuel_type == 'Petrol':
            fuel_type = 1
        else:
            fuel_type = 0
        transmission = request.form['Transmission']
        if transmission == 'Manual':
            transmission = 1
        else:
            transmission = 0

        cols = ([[Present_Price, Kms_Driven2, fuel_type, seller_type, transmission, second_hand]])
        cols = np.array(cols)
        features = cols
        print(features)
        predictions = car_model.predict(features)
        output = round(predictions[0], 2)
        print(output)
        if output <= 0:
            return render_template('./output.html',prediction_text="Cannot Sell")
        else:
            return render_template('./output.html', prediction_text=f"You can sell at {output:.2f} lacs. ")
    else:
        return render_template('./home.html')


if __name__ == "__main__":
    app.run(debug=True)
