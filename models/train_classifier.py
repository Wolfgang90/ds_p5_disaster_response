import sys
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(database_filepath):
    """
        Loads dataframe from database
        Input:
            database_filepath: Filepath of the database to be loaded
        Output:
            X (pd.Series): Series of text messages
            Y (pd.DataFrame): DataFrame with target categories
            category_names (list): List with target category names
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('disaster', engine)
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'],axis=1)
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """
        Transforms text into cleaned tokens
        Input:
            text (str): Text to be tokenized
        Output:
            cleaned_tokens (list): List of cleaned tokens
    """
    #tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
        Builds pipeline-model
        Output:
            pipeline-model
    """
    pipeline = Pipeline([
    ('text_pipeline', Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer())
    ])),
    
    ('clf', MultiOutputClassifier(RandomForestClassifier(max_depth=None, min_samples_leaf=1, min_samples_split=2)))
    ])    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
        Prints f1 score, precision and recall for the test set for each category
        Input:
            model (sklearn classifier): trained model
            X_test (pd.Series): X-values of test set
            Y_test (pd.DataFrame): Y-values of test set
            category_names (list): List with target category names
    """
    Y_pred = model.predict(X_test)
    for i, col in enumerate(category_names):
        classification_report(Y_test[col], Y_pred[:,i:i+1])
        print("Accuracy score for \'{}\': {:.4f}".format(col, accuracy_score(Y_test.values[:,i], Y_pred[:,i])))
        print("Classification report \'{}\':\n {}".format(col,classification_report(Y_test.values[:,i], Y_pred[:,i], target_names=["0","1"])))


def save_model(model, model_filepath):
    pass


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