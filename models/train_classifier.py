import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sqlite3
import sqlalchemy
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load data from SQLite database.
    
    Input:
    database_filepath: File path of the database
    
    Output:
    X: Features
    Y: Target
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    Tokenize and lemmatize the sentence into individual words
    Input:
        text: the whole sentence
    
    Output:
        list of individual tokens
    """
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
   
    return [lemmatizer.lemmatize(t).lower().strip() for t in tokens]


def build_model():
    """
    Build classifier and tune model using GridSearchCV.
    
    Output:
    cv: classifier 
    """    
    pipeline = Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', MultiOutputClassifier(RandomForestClassifier()))
                ])
                    
    parameters = {
        'clf__estimator__n_estimators' : [20, 30],
        'clf__estimator__min_samples_leaf': [1,2]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the performance of model and returns classification report. 
    
    Input:
        model: classifier
        X_test: features of test dataset
        Y_test: targets for test data in X_test
    
    Output:
        Classification report for each column
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))
        
        
def save_model(model, model_filepath):
    """ Exports the model as a pickle file."""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """ Builds the model, trains the model, evaluates the model, saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
