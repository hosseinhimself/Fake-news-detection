from flask import Flask, request, jsonify
from models import XGBoostModel, RandomForestModel, PyCaretModel

app = Flask(__name__)

# Instantiate models
xgboost_model = XGBoostModel()
random_forest_model = RandomForestModel()
pycaret_model = PyCaretModel()

# Load pre-trained models
xgboost_model.load_model('xgboost_model.pkl')
random_forest_model.load_model('random_forest_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()

    # Extract text data and algorithm choice
    title = data['title']
    author = data['author']
    text = data['text']
    algorithm = data['algorithm']

    # Make predictions based on the chosen algorithm
    if algorithm == 'xgboost':
        prediction = xgboost_model.predict(title, author, text)
    elif algorithm == 'random_forest':
        prediction = random_forest_model.predict(title, author, text)
    elif algorithm == 'pycaret':
        prediction = pycaret_model.predict(title, author, text)
    else:
        return jsonify({'error': 'Invalid algorithm choice'})

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
