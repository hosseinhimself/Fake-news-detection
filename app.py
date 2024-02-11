from flask import Flask, request, jsonify, render_template
from models import XGBoostModel, RandomForestModel, PyCaretModel

app = Flask(__name__)

# Instantiate models
xgboost_model = XGBoostModel()
random_forest_model = RandomForestModel()
pycaret_model = PyCaretModel()

# Load pre-trained models
xgboost_model.load_model('xgboost_model.pkl')
random_forest_model.load_model('random_forest_model.pkl')

# List of algorithms for dropdown
algorithms = ['xgboost', 'random_forest', 'pycaret']


@app.route('/')
def index():
    return render_template('index.html', algorithms=algorithms)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get data from the HTML form
        title = request.form['title']
        author = request.form['author']
        text = request.form['text']
        algorithm = request.form['algorithm']

        # Make predictions based on the chosen algorithm
        if algorithm == 'xgboost':
            prediction = xgboost_model.predict(title, author, text)
        elif algorithm == 'random_forest':
            prediction = random_forest_model.predict(title, author, text)
        elif algorithm == 'pycaret':
            prediction = pycaret_model.predict(title, author, text)
        else:
            return jsonify({'error': 'Invalid algorithm choice'})

        return render_template('result.html', prediction=prediction)

    # If GET method, render the form
    return render_template('index.html', algorithms=algorithms)


if __name__ == '__main__':
    app.run(debug=True)
