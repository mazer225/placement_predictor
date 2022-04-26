from flask import Flask, request, jsonify
import numpy as np
import pickle

model1 = pickle.load(open('modelNB.pkl', 'rb'))
model2 = pickle.load(open('modelRF.pkl', 'rb'))
model3 = pickle.load(open('modelSVC.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return "Hello world"


@app.route('/predict', methods=['POST'])
def predict():
    cgpa = request.form.get('cgpa')
    iq = request.form.get('iq')
    profile_score = request.form.get('profile_score')

    input_query = np.array([[cgpa, iq, profile_score]])

    result1 = model1.predict(input_query)[0]
    result2 = model2.predict(input_query)[0]
    result3 = model3.predict(input_query)[0]

    final_result_list = []
    final_result_list.extend([result1, result2, result3])

    print(final_result_list)

    one_count = final_result_list.count(1)
    zero_count = final_result_list.count(0)
    if(one_count > zero_count):
        result = 1
    else:
        result = 0

    return jsonify({'placement': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
