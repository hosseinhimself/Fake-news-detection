import os
import pickle
import pandas as pd
import xgboost as xgb
from pycaret.classification import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


class Model:
    def __init__(self):
        self.model = None
        self.vectors = None
        self.train_data = None
        self.test_data = None
        self.submit_data = None
        self.TF_IDF = None

    def load_data(self, train_path, test_path, submit_path):
        # Load training, testing, and submission datasets
        self.train_data = pd.read_csv(train_path)
        self.test_data = pd.read_csv(test_path)
        self.submit_data = pd.read_csv(submit_path)

    def save_tfidf(self):
        try:
            # Save the trained model using pickle
            with open('tf_idf.pkl', 'wb') as model_file:
                pickle.dump(self.TF_IDF, model_file)

            print(f"TF_IDF vectorizer saved successfully!")

        except Exception as e:
            print(f"Saving failed! {str(e)}")

    def load_tfidf(self):
        try:
            # Load a pre-trained model from a file
            with open('tf_idf.pkl', 'rb') as model_file:
                self.TF_IDF = pickle.load(model_file)

            print(f"TF_IDF loaded successfully!")

        except Exception as e:
            print(f"Loading failed! {str(e)}")

    def embedding_data(self, df):
        # Process and transform text data using TF-IDF vectorizer
        df['title'].fillna('Untitled', inplace=True)
        df['author'].fillna('unknown', inplace=True)
        df.dropna(subset=['text'], inplace=True)

        df['combined'] = df['title'].astype(str) + ' ' + df['author'].astype(str) + ' ' + df['text'].astype(str)

        self.TF_IDF = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=0.01)

        res = self.TF_IDF.fit_transform(df['combined'])
        self.save_tfidf()

        return res

    def save_model(self, filename):
        try:
            # Save the trained model using pickle
            with open(filename, 'wb') as model_file:
                pickle.dump(self.model, model_file)

            print(f"Model {type(self).__name__} saved successfully!")

        except Exception as e:
            print(f"Saving failed! {str(e)}")

    def load_model(self, filename):
        try:
            # Load a pre-trained model from a file
            with open(filename, 'rb') as model_file:
                self.model = pickle.load(model_file)

            print(f"Model {type(self).__name__} loaded successfully!")

        except Exception as e:
            print(f"Loading failed! {str(e)}")


class XGBoostModel(Model):
    def __init__(self):
        super().__init__()

    def build_model(self, train_path, test_path):

        print("Building XGBoost model stared!")

        # Build an XGBoost model using training data
        self.load_data(train_path, test_path, 'dataset/submit.csv')
        X = self.embedding_data(self.train_data)
        y = self.train_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Set XGBoost hyperparameters
        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'eta': 0.3,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }

        num_rounds = 100
        self.model = xgb.train(params, dtrain, num_rounds)

        y_pred = self.model.predict(dtest)

        # Print classification report and accuracy
        print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy:.2f}")

        # Save the trained XGBoost model
        self.save_model('xgboost_model.pkl')

    def predict(self, title, author, text):
        # Make predictions using the trained XGBoost model
        text = f'{title} {author} {text}'

        if self.TF_IDF is None:
            self.load_tfidf()

        x = self.TF_IDF.transform([text])

        if self.model is None:
            model_file_path = 'xgboost_model.pkl'
            if not os.path.exists(model_file_path):
                raise Exception("Model file does not exist. Please build the model first.")

            self.load_model(model_file_path)

        dm = xgb.DMatrix(x)

        if self.model.predict(dm) == 1:
            return True
        else:
            return False


class RandomForestModel(Model):
    def __init__(self):
        super().__init__()

    def build_model(self, train_path, test_path):

        print("Building RandomForest model stared!")

        # Build a Random Forest model using training data
        self.load_data(train_path, test_path, 'dataset/submit.csv')
        X = self.embedding_data(self.train_data)
        y = self.train_data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Set Random Forest hyperparameters
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        # Print classification report and accuracy
        print(classification_report(y_test, y_pred))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Overall Accuracy: {accuracy:.2f}")

        # Save the trained Random Forest model
        self.save_model('random_forest_model.pkl')

    def predict(self, title, author, text):
        # Make predictions using the trained Random Forest model
        text = f'{title} {author} {text}'

        if self.TF_IDF is None:
            self.load_tfidf()

        x = self.TF_IDF.transform([text])

        if self.model is None:
            model_file_path = 'random_forest_model.pkl'
            if not os.path.exists(model_file_path):
                raise Exception("Model file does not exist. Please build the model first.")

            self.load_model(model_file_path)

        res = self.model.predict(x)

        if res == 1:
            return True
        else:
            return False


class PyCaretModel(Model):
    def __init__(self):
        super().__init__()

    def build_model(self, train_path, test_path):
        print("Building PyCaret model started!")

        # Load data
        train = pd.read_csv(train_path)
        train = train[['title', 'author', 'text', 'label']]  # Select relevant columns
        test = pd.read_csv(test_path)

        # Set up PyCaret
        clf = setup(data=train, target='label', session_id=42, log_experiment=True,
                    experiment_name='pycaret_experiment')

        # Compare models and select the best
        self.model = compare_models()

        # Save the best model
        save_model(self.model, 'pycaret_model')

        print("Building PyCaret model completed!")

    def predict(self, title, author, text):
        print("Making predictions using PyCaret model...")

        # Load the saved PyCaret model
        if self.model is None:
            if self.model is None:
                # Check if the model file exists
                model_file_path = 'pycaret_model.pkl'
                if not os.path.exists(model_file_path):
                    raise Exception("Model file does not exist. Please build the model first.")

                self.model = load_model('pycaret_model')

        # Make predictions on new data
        new_data = pd.DataFrame({'title': [title], 'author': [author], 'text': [text]})
        prediction = predict_model(self.model, data=new_data)

        if prediction['prediction_label'].iloc[0] == 0:
            return False
        else:
            return True


if __name__ == '__main__':
    # Sample input for prediction
    title = 'Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times'
    author = 'David Streitfeld'
    text = """
    If at first you don’t succeed, try a different sport. Tim Tebow, who was a Heisman   quarterback at the University of Florida but was unable to hold an N. F. L. job, is pursuing a career in Major League Baseball. He will hold a workout for M. L. B. teams this month, his agents told ESPN and other news outlets. “This may sound like a publicity stunt, but nothing could be further from the truth,” said Brodie Van Wagenen,   of CAA Baseball, part of the sports agency CAA Sports, in the statement. “I have seen Tim’s workouts, and people inside and outside the industry  —   scouts, executives, players and fans  —   will be impressed by his talent. ” It’s been over a decade since Tebow, 28, has played baseball full time, which means a comeback would be no easy task. But the former major league catcher Chad Moeller, who said in the statement that he had been training Tebow in Arizona, said he was “beyond impressed with Tim’s athleticism and swing. ” “I see bat speed and power and real baseball talent,” Moeller said. “I truly believe Tim has the skill set and potential to achieve his goal of playing in the major leagues and based on what I have seen over the past two months, it could happen relatively quickly. ” Or, take it from Gary Sheffield, the former   outfielder. News of Tebow’s attempted comeback in baseball was greeted with skepticism on Twitter. As a junior at Nease High in Ponte Vedra, Fla. Tebow drew the attention of major league scouts, batting . 494 with four home runs as a left fielder. But he ditched the bat and glove in favor of pigskin, leading Florida to two national championships, in 2007 and 2009. Two former scouts for the Los Angeles Angels told WEEI, a Boston radio station, that Tebow had been under consideration as a high school junior. “’x80’x9cWe wanted to draft him, ’x80’x9cbut he never sent back his information card,” said one of the scouts, Tom Kotchman, referring to a questionnaire the team had sent him. “He had a strong arm and had a lot of power,” said the other scout, Stephen Hargett. “If he would have been there his senior year he definitely would have had a good chance to be drafted. ” “It was just easy for him,” Hargett added. “You thought, If this guy dedicated everything to baseball like he did to football how good could he be?” Tebow’s high school baseball coach, Greg Mullins, told The Sporting News in 2013 that he believed Tebow could have made the major leagues. “He was the leader of the team with his passion, his fire and his energy,” Mullins said. “He loved to play baseball, too. He just had a bigger fire for football. ” Tebow wouldn’t be the first athlete to switch from the N. F. L. to M. L. B. Bo Jackson had one   season as a Kansas City Royal, and Deion Sanders played several years for the Atlanta Braves with mixed success. Though Michael Jordan tried to cross over to baseball from basketball as a    in 1994, he did not fare as well playing one year for a Chicago White Sox minor league team. As a football player, Tebow was unable to match his college success in the pros. The Denver Broncos drafted him in the first round of the 2010 N. F. L. Draft, and he quickly developed a reputation for clutch performances, including a memorable   pass against the Pittsburgh Steelers in the 2011 Wild Card round. But his stats and his passing form weren’t pretty, and he spent just two years in Denver before moving to the Jets in 2012, where he spent his last season on an N. F. L. roster. He was cut during preseason from the New England Patriots in 2013 and from the Philadelphia Eagles in 2015.
    """
    # Instantiate models
    xgboost_model = XGBoostModel()
    random_forest_model = RandomForestModel()
    pycaret_model = PyCaretModel()

    # Build models
    #xgboost_model.build_model('dataset/train.csv', 'dataset/test.csv')
    #random_forest_model.build_model('dataset/train.csv', 'dataset/test.csv')

    # Predictions using XGBoost model
    xgboost_prediction = xgboost_model.predict(title, author, text)
    print(f"XGBoost Prediction: {xgboost_prediction}")

    # Predictions using Random Forest model
    random_forest_prediction = random_forest_model.predict(title, author, text)
    print(f"Random Forest Prediction: {random_forest_prediction}")

    #pycaret_model.build_model('dataset/train.csv', 'dataset/test.csv')


    # Predictions using PyCaret model
    pycaret_prediction = pycaret_model.predict(title, author, text)
    print(f"PyCaret Prediction: {pycaret_prediction}")
