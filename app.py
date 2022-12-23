from flask import Flask, render_template, request

import pickle

import pandas as pd

from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
BHP_df = pd.read_csv('./Cleaned_Data_Backup.csv')
pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

@app.route('/',methods=['GET'])
def index():
    locations = sorted(BHP_df['location'].unique())
    return render_template('index1.html',locations = locations)

standard_to = StandardScaler()

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bath = request.form.get('bath')
        sqft = request.form.get('total_sqft')
    
        #print(location, bhk, bath, sqft)
    
        input = pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','bhk'])
        prediction = pipe.predict(input)[0]
        output=round(prediction,2)
        if output<0:
            return render_template('index1.html',prediction_text="Sorry you cannot Buy this type of House")
        else:
            return render_template('index1.html',prediction_text="You Can Buy This Type Of House is Lakh {}".format(output))
    else:
        return render_template('index1.html')

if __name__=="__main__":
    app.run(debug=True,port=5003)
