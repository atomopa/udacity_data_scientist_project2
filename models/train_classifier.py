import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - filepath to the database to be read
    
    OUTPUT:
    X - X matrix
    Y - Y matrix
    category_names - column headers from Y matrix
    
    This function reads the provided database into X and Y matrices.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql("SELECT * FROM data", engine)

    X = df['message'].values
    Y = df.iloc[:,4:].values

    category_names = (df.iloc[:,4:]).columns.values
    
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT:
    text - string to be tokenized
    
    OUTPUT:
    clean_tokens - list of tokens
    
    This function tokenizes the provided string using word_tokenize and WordNetLemmatizer as well as converts to lower case and strips spaces.
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    OUTPUT:
    pipeline - ready to use ML pipeline
    
    This function creates a ML pipeline.
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - trained model
    X_test - input matrix of test dataset
    Y_test - results matrix of test dataset
    category_names - category names (Y_test column headers)
    
    This function predicts the answers to X_test and compares the prediction to Y_test to evaluate the model performance.
    '''
    Y_pred = model.predict(X_test)
    
    tp = np.zeros(36)
    fp = np.zeros(36)
    fn = np.zeros(36)
    for i, entry in enumerate(Y_test):
        for j in range(36):
            if Y_test[i][j] == 1 and Y_pred[i][j] == 1:
                tp[j] = tp[j] + 1
            elif Y_test[i][j] == 0 and Y_pred[i][j] == 1:
                fp[j] = fp[j] + 1
            elif Y_test[i][j] == 1 and Y_pred[i][j] == 0:
                fn[j] = fn[j] + 1

    precision = [tp[i]/(tp[i]+fp[i]) for i in range(36)]
    recall = [tp[i]/(tp[i]+fn[i]) for i in range(36)]
    f1_score = [2*(precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(36)]

    df = pd.DataFrame(list(zip(precision, recall, f1_score)), columns=['precision', 'recall', 'f1_score'], index=category_names)

    print (df)


def save_model(model, model_filepath):
    '''
    INPUT:
    model - trained model
    model_filepath - path where model will be saved
    
    This function pickles the model and stores it.
    '''
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        # load the data from sql into X and Y and category_names
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        # build the model
        print('Building model...')
        model = build_model()
        
        # train the model
        print('Training model...')
        model.fit(X_train, Y_train)
        
        # evaluate the model
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        # save the model
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
