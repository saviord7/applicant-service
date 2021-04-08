from json import dumps

from flask import Flask, request
from flask import make_response
from flask_cors import CORS
import numpy as np
import data_processing
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)
CORS(app)


@app.route('/api/predict', methods=['POST'])
def predict():
    json = request.get_json(force=True)
    data_raw = json['data']
    data=[]
    
    for i in data_processing.arr_field_names:
        if not i in data_raw:
          data_raw[i] = 0
          
        print(i)

        t = 0
        try:
          t = int(data_raw[i])
        except ValueError:
          
          
        data.append(t)
        
    print(data)
    
    prediction = np.array(data_processing.predict_list(data))
    response = make_response('{"prediction": ' + dumps(prediction.tolist()) + '}')
    response.headers['Content-Type'] = "application/json"
    return response


if __name__ == "__main__":
    app.run(debug=True)