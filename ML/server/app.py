from json import dumps
from flask import Flask, request
from flask import make_response
from flask_cors import CORS
import numpy as np
import data_processing


app = Flask(__name__)
CORS(app)


@app.route('/api/predict', methods=['POST'])
def predict():
    json = request.get_json(force=True)
    data = json['data']
    prediction = np.array(data_processing.predict_list(data))
    response = make_response('{"prediction": ' + dumps(prediction.tolist()) + '}')
    response.headers['Content-Type'] = "application/json"
    return response


if __name__ == "__main__":
    app.run(debug=True)