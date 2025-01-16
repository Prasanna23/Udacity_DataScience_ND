# %% [markdown]
# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# %%
# import libraries
import pandas as pd
from sqlalchemy import create_engine 
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# %%
from sklearn.model_selection import GridSearchCV


# %%
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#nltk.download()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
# load data from database
engine = create_engine('sqlite:///Disaster_Database.db')
df = pd.read_sql_table('Disaster_Msg',engine)
df.head()



# %%
df['message'] = df['message'].fillna('')  # Replace NaN with empty string
df.iloc[:,4:] = df.iloc[:,4:].fillna(0)
    
# Convert to string
df['message'] = df['message'].astype(str)
X = df['message']
Y = df.iloc[:,4:]

# %% [markdown]
# ### 2. Write a tokenization function to process your text data

# %%
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

# %%
sample_text = "Disaster response is crucial for effective emergency management!"
processed_tokens = tokenize(sample_text)
print(processed_tokens)

# %% [markdown]
# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# %%
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

# %% [markdown]
# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# %%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# %%

#sklearn.set_config(enable_metadata_routing=True)
pipeline.fit(X_train, y_train)


# %%
print(X_train)

# %%
print(y_train)

# %% [markdown]
# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# %%
y_pred = pipeline.predict(X_test)


# %%
def evaluate_model(y_test, y_pred, y_columns):
    # Comprehensive evaluation
    print("Detailed Classification Report:")
    for i, col in enumerate(y_columns):
        print(f"\nMetrics for {col}:")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

# %%
def classification_rep_df(y_test,y_pred,y_columns,name):
    for i, col in enumerate(y_columns):
        #print(f"\nMetrics for {col}:")
        globals()[f'df_{name}_{col}'] = pd.DataFrame(classification_report(y_test.iloc[:, i], y_pred[:, i], digits=2,
                                        output_dict=True)).T

# %%
evaluate_model(y_test, y_pred, Y.columns)


# %%
pipeline.get_params().keys()

# %% [markdown]
# ### 6. Improve your model
# Use grid search to find better parameters. 

# %%
parameters =  { 
   'classifier__estimator__n_estimators': [25, 50,100,150],
   'classifier__estimator__max_depth': [3, 6, 9], 
    'classifier__estimator__max_leaf_nodes': [3, 6, 9], 
} 

cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='accuracy', cv=2, n_jobs=-1)


# %%
cv.fit(X_train, y_train)


# %%
print("Best Hyperparameters:", cv.best_params_)


# %% [markdown]
# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# %%
tuned_pipeline = Pipeline([
        # TF-IDF Vectorization with custom tokenizer
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize,
            max_features=5000,
            ngram_range=(1, 2)
        )),
        # Multi-output classifier with RandomForest
        ('classifier', MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=25, 
                random_state=42, 
                max_depth=3,
                max_leaf_nodes=3,
                n_jobs=-1
            )
        ))
    ])

# %% [markdown]
# Fit the tuned pipeline and predict using the same. Evaluate the model after that.

# %%
tuned_pipeline.fit(X_train, y_train)
y_pred_tuned = tuned_pipeline.predict(X_test)

# %%
evaluate_model(y_test, y_pred_tuned, Y.columns)

# %%
tuned_rep_df = pd.DataFrame()
classification_rep_df(y_test,y_pred_tuned,Y.columns,'tuned')
classification_rep_df(y_test,y_pred,Y.columns,'beftuned')

# %%
#Heat map of precision, recall and f1-score for "related" variable
# model without tuned parameters
df_beftuned_related['support'] = df_beftuned_related.support.apply(int)

df_beftuned_related.style.background_gradient(cmap='viridis',
                             subset=pd.IndexSlice['0':'9', :'f1-score'])

# %%
#Heat map of precision, recall and f1-score for "related" variable
# model with tuned parameters
df_tuned_related['support'] = df_tuned_related.support.apply(int)

df_tuned_related.style.background_gradient(cmap='viridis',
                             subset=pd.IndexSlice['0':'9', :'f1-score'])

# %% [markdown]
# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# %%
from sklearn.dummy import DummyClassifier

pipeline_dummy = Pipeline([
        # TF-IDF Vectorization with custom tokenizer
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize,
            max_features=5000,
            ngram_range=(1, 2)
        )),
        # Multi-output classifier with RandomForest
        ('classifier', MultiOutputClassifier(
                    DummyClassifier(strategy='most_frequent')
            )
        )
    ])

# %%
pipeline_dummy.fit(X_train,y_train)


# %%
y_pred_dummy = pipeline_dummy.predict(X_test)

# %%
evaluate_model(y_test,y_pred_dummy,Y.columns)

# %%
classification_rep_df(y_test,y_pred_dummy,Y.columns,'dummy')

# %%
#Heat map of precision, recall and f1-score for "related" variable
# model with tuned parameters
df_dummy_related['support'] = df_dummy_related.support.apply(int)

df_dummy_related.style.background_gradient(cmap='viridis',
                             subset=pd.IndexSlice['0':'9', :'f1-score'])

# %% [markdown]
# After looking at the performance metrics, It's evident that Random Forest model is doing better than dummy model. We will use the initial model for our classification purposes.

# %% [markdown]
# ### 9. Export your model as a pickle file

# %%
import pickle

# save the model into pickle file
with open('RF_model.pkl','wb') as f:
    pickle.dump(pipeline,f)


# %% [markdown]
# ### 10. Use this notebook to complete `train_classifier.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# %%



