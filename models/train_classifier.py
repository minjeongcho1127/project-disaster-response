import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix

import pickle

def load_data(database_filepath):
    """
    Load data from the database
    
    Input:
        database_filepath: path to the database (*.db) file
        
    Output:
        X: messages
        y: categories for each message
        category_names: list of the category names
    """
    
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('DisasterResponseTable', engine)
    
    # replace the maximum value of 2 in the column 'related' to 1
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    # drop the 'child_alone' column which only contains zeros.
    df.drop('child_alone', axis = 1, inplace = True)
    
    # Using the message column (X), predict classifications for 35 categories (y)
    X = df['message']
    y = df.iloc[:, 4:]
    category_names = y.columns
    
    return X, y, category_names

def tokenize(text):
    """
    A tokenization function to process the text data
    
    Input:
        text: original text message that needs to be tokenized
    
    Output:
        clean_tokens: list of tokens processed from the text input
    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens

def build_model(clf = AdaBoostClassifier()):
    """
    Build a machine learning pipeline
    
    Input:
        clf: classifier model (default = AdaBoostClassifier())
    
    Output:
        cv: a machine learning pipeline after GridSearchCV
    """

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
        ])),
        ('classifier', MultiOutputClassifier(clf))
    ])

    parameters = {'classifier__estimator__learning_rate': [0.1, 0.5, 1.0],
              'classifier__estimator__n_estimators': [10, 20, 50]}

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = -1, verbose = 3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Split data into train and test sets and then train and test the model
    
    Input:
        model: a machine learning model
        X_test: test messages
        Y_test: categories according to each test message
        category_names: category names according to each category
    
    Output:
        None: print classification_report
    """
    Y_pred = model.predict(X_test)
    
    print(classification_report(Y_test.values, Y_pred, target_names = category_names))

def save_model(model, model_filepath):
    """
    Export the trained model as a pickle file
    
    Input:
        model: the trained machine learning model
        model_filepath: filepath for the *.pkl file
    
    Output:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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