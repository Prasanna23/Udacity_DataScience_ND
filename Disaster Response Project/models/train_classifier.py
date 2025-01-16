import sys
import sqlalchemy
import re
from sqlalchemy import create_engine 
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    #engine = create_engine('sqlite:///Disaster_Database.db')
    df = pd.read_sql_table('Disaster_Msg',engine)
    df['message'] = df['message'].fillna('')  # Replace NaN with empty string
    df.iloc[:,4:] = df.iloc[:,4:].fillna(0)
    df['message'] = df['message'].astype(str)
    X = df['message']
    Y = df.iloc[:,4:]   
    category_names = Y.columns # This will be used for visualization purpose
    print(type(X),type(Y),type(category_names))
    return X, Y, category_names


def tokenize(text):
            # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Lemmatization
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Remove very short tokens
    tokens = [token for token in tokens if len(token) > 1]
    
    return tokens


def build_model():
    pipeline = Pipeline([
        # TF-IDF Vectorization with custom tokenizer
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize,
            max_features=5000,
            ngram_range=(1, 2)
        )),
        # Multi-output classifier with RandomForest
        ('classifier', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            )
        ))
    ])
    return pipeline


def evaluate_model(Y_test, Y_pred, y_columns):
    print("Detailed Classification Report:")
    for i, col in enumerate(y_columns):
        print(f"\nMetrics for {col}:")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        print(model)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('...predicting..')
        y_pred = model.predict(X_test)
        
        print('Evaluating model...')
        evaluate_model(Y_test, y_pred, Y.columns)

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