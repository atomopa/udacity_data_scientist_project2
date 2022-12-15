import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - filepath the the message data
    categories_filepath - filepath to the categories data
    
    OUTPUT:
    df - dataframe containing the contents of the csv files
    
    This function reads in the messages and categories csv files and combines them to a dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    '''
    INPUT:
    df - dataframe to be cleaned
    
    OUTPUT:
    df - cleaned dataframe 
    
    This function cleans the dataframe by splitting the categories columns and converting them to binary values
    '''
        
    categories = df['categories'].str.split(';', expand=True)
    
    row = categories.loc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop rows where related = 2
    categories = categories[categories.related != 2]
    
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - dataframe to be saved
    database_filename - database path
    
    This function stores the dataframe to a database
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('data', engine, index=False, if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        # load the csv files into dataframe
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        # clean dataframe
        print('Cleaning data...')
        df = clean_data(df)
        
        # store dataframe
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
