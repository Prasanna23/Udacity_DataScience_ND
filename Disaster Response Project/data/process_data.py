import sys
import pandas as pd
from sqlalchemy import create_engine 



def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    messages.head()
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    categories.head()
    df = pd.merge(messages,categories, on="id")
    categories = categories['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[1]
    category_colnames = [item[:-2] for item in row]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories])
    return df


def clean_data(df):
    duplicates = df.duplicated()
    duplicate_rows = df[duplicates]
    # drop duplicates
    df_dedup = df.drop_duplicates() 
    # check number of duplicates
    print('Number of duplicates')
    print(df.shape[0] - df_dedup.shape[0])
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster_Msg', engine, if_exists='replace',index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning categories data ...')
        df = clean_data(df)
        
        print('Saving data to SQLite DB : {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data has been saved to database!')
    else: # Print the help message so that user can execute the script with correct parameters
        print("Please provide the arguments correctly: \nSample Script Execution:\n\> python process_data.py disaster_messages.csv disaster_categories.csv disaster_response_db.db \n\
Arguments Description: \n\
1) Path to the CSV file containing messages (e.g. disaster_messages.csv)\n\
2) Path to the CSV file containing categories (e.g. disaster_categories.csv)\n\
3) Path to SQLite destination database (e.g. disaster_response_db.db)") 

if __name__ == '__main__':
    main()