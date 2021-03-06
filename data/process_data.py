import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load 'messages.csv' and 'categories.csv' files and merge the messages and categories datasets
    
    Input:
        messages_filepath: path to 'messages.csv'
        categories_filepath: path to 'categories.csv'
    
    Output:
        df: merged dataset containing messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')

    return df
    

def clean_data(df):
    """
    Clean up the categories in the dataset
    
    Input:
        df: merged dataset containing messages and categories
        
    Output:
        df: merged dataset containing messages and the cleaned up categories
    """
    
    # create a dataframe of the 35 individual category columns
    categories = df['categories'].str.split(pat=';', expand = True)
    
    # rename the column names of 'categories'
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # replace 'categories' column in df with new category columns
    # step 1. drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    # step 2. concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # remove duplicates
    df.drop_duplicates(inplace = True)

    # replace the maximum value of 2 in the column 'related' to 1
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    # drop the 'child_alone' column which only contains zeros.
    df.drop('child_alone', axis = 1, inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Save the clean dataset into an sqlite database
    
    Input:
        df: clean dataset
        database_filename: filename of an sqlite database
        
    Output:
        None
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists = 'replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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