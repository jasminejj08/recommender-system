## This script cleans the input data

## the input data (for ratings) is separated into 7 csv files that have 
# the following structure:
# ex. user_rating_0_to_1000.csv: ID, Name, Rating
## where ID is the USER ID, Name is the BOOK Name, and Rating is based on the
## following scale (following Goodreads rating system):
# NaN: No Rating --> adjusted in this cleaning process
# 1: didn't like it 
# 2: it was ok
# 3: liked it
# 4: really liked it
# 5: it was amazing
# NaN: This user doesn't have any rating (refer to ex. user_rating_2000_to_3000.csv)

# refer to the following link for clarification on the rating system:
#https://www.goodreads.com/topic/show/17895147-what-is-your-rating-system-on-goodreads

## there are 23 books metadata csv files
## important columns include: Id (book ID), Name (book name)
## note that it's 'Id' for books.csv files and 'ID' for user rating csv files

## The goal of this cleaning process is to:
# 1. Combine all 7 csv files into a single dataframe
# 2. Standardize the ratings into a consistent numerical scale (1-5)
# 3. Handle missing values (NaN ratings)

## Note: can alter the amount of data used by commenting out some file paths in the main function

## Note: this file must be run first (before any other scripts)

## Note: any code that says "## for debugging purposes" can be uncommented to see
## running outputs; these are mainly for my own understanding and debugging of the code


import pandas as pd
import numpy as np
#import os
from pathlib import Path

def clean_data(file_paths, book_file_paths, rating_scale):
    """
    Preprocess and clean the user rating data from multiple CSV files
    
    inputs
    file_paths: list of file paths to the input CSV files containing user ratings
    book_file_paths: list of file paths to the book metadata CSV files
    rating_scale: dictionary mapping original rating descriptions to numerical values

    output: cleaned dataframe with structure: user_id, book_id, rating; all numbers

    order of operations: 
    1. load each book metadata file (ex. books1-100k.csv) to create a mapping dictionary
       from book name to book ID
    2. load each ratings csv file into a dataframe, clean the data
        (remove duplicates, missing values, change ratings to numerical values, change book name to book ID)
    3. combine all cleaned dataframes into a single dataframe
    """


    ## note the structure of each column in the ratings.csv files; need to change the
    ## 'Name' (book name) column to book ID for utility matrix later
    ## loading all books metadata files to create a mapping dictionary
    
    ## need to convert the books names to its book ID for consistency and to make the
    ## utility matrix
    ## use the ex. books1-100k.csv file to create another mapping dictionary (csv file containing metadata of books)
    ## -- this file must be in the same directory as this script
    ## Note: there are 23 books.csv files in total (books1-1000.csv to books4000-5000.csv)
    ## thus, all 23 files are loaded and combined to create a single df, from which the mapping dictionary is created
    ## -- this may take some time to run

    all_book_files = [] ## empty list to hold all book dataframes
    for book_file in book_file_paths: ## go through each book metadata file (see main function for list of files)
        try:
            temp_df = pd.read_csv(book_file) ## read each file into a dataframe
        except FileNotFoundError:
            print(f"WARNING: {book_file} not found")
            continue
        all_book_files.append(temp_df) ## append to the list
    books_df = pd.concat(all_book_files, ignore_index=True) ## combine all book dataframes into a single dataframe

    ## for debugging purposes
    # print("Combined books metadata preview:")
    # print(books_df.head(5))
    # print(f'total length of concatenated books df: {len(books_df)}')
    # print("\n")

    ## cleaning process for books_df dataframe
    ## make sure there are no duplicate book names in the books_df dataframe that would result in
    ## two different book IDs for the same book --> make all book names lowercase and strip leading/trailing spaces
    books_df['unique_name'] = books_df['Name'].str.lower().str.strip()
    books_df = books_df.drop_duplicates(subset=['unique_name'])

    ## FOR CONTENT BASED; save the cleaned books_df to a csv file --> contains metadata
    ## used to determine content similarities for user
    books_metadata = books_df[['Id', 'unique_name', 'Authors']].copy()
    books_metadata = books_metadata.rename(columns={'Authors': 'authors'})
    books_metadata['authors'] = books_metadata['authors'].str.lower().str.strip()
    books_metadata.to_csv('Cleaned_Data_Output/cleaned_books_metadata.csv.gz', index=False, compression='gzip')
    print("cleaned books metadata saved; use for content-based rec")
    # print(f'book metadata shape: {books_metadata.shape}\n')

    ## create a mapping dictionary from book name to book ID
    #book_name_id_map = pd.Series(books_df.Id.values, index=books_df.Name).to_dict()
    book_name_id_map = dict(zip(books_df['unique_name'], books_df['Id'])) ## joining the columns using zip then converting to a dict

    ## for debugging purposes
    #print("Book name to ID mapping preview:")
    ##print(list(book_name_id_map.items())[:5])
    ##print("\n")

    ## initialize an empty list to hold all the dataframes for the rating files; will append to this
    merged_df_list = []

    ## iterate through each file path for the ratings csv files
    for file_path in file_paths:

        ## for debugging purposes
        #print(f"current file: {file_path}")

        ## read the csv file into a dataframe
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"WARNING: {file_path} not found")
            continue

        ## debugging
        #print(f'\nbefore cleaning {file_path}, length: {len(df)}')

        ## debugging
        ## display the first few lines of each file to understand the structure
        # print(f"Preview of {file_path}:")
        # print(df.head(5))
        # print("\n")

        ## first strip any leading/trailing spaces in the Name column (book name); will
        ## need this to correctly map the book ID based on the mapping created earlier
        df['Name'] = df['Name'].str.lower().str.strip()
        ## preparing for mapping to book ID
        #df['unique_title'] = df['Name'].str.lower().str.strip()

        ## change the 'Name' column (book name) to book ID using the mapping dictionary
        df['book_id'] = df['Name'].map(book_name_id_map)
        ## this will result in NaN values for any book names not found in the mapping dictionary
        ## drop these rows bc it is not useful data
        df = df.dropna(subset=['book_id'])


        ## clean the rating column too as its text
        df['Rating'] = df['Rating'].str.lower().str.strip()

        ## change the 'Rating' column to numerical values based on rating scale dictionary
        ## visible in the main function
        df['rating'] = df['Rating'].map(rating_scale)

        ## Note: we may have NaN values in the 'rating' column, which indicates no rating given by user
        ## KEEP THIS ROWS --> want to predict ratings for these books later on

        ## debugging
        # print(f'\nRating value counts for {file_path}:\n{df["rating"].value_counts(dropna=False)}\n')

        ## remove any duplicate entries (same user ID AND book name; aka multiple ratings for same book by same user))
        ## keeping the first occurence; can also consider averaging ratings if needed (another approach)
        df = df.drop_duplicates(subset=['ID', 'book_id'])

        ## remove any rows with book = 'Rating' (not useful data)
        ## this is a manual fix for some csv files that have this issue
        df = df[df['Name'] != 'rating']

        ## one final check to ensure ratings are within the range or NaN; dropping any
        ## rows that don't meet this criteria
        valid_ratings = [1, 2, 3, 4, 5, np.nan]
        df = df[df['rating'].isin(valid_ratings)]


        ## utility matrix will require user_id, book_id, rating columns;
        ## keep only these columns and append it to the merged_df_list list
        df = df[['ID', 'book_id', 'rating']].rename(columns={'ID': 'user_id'})        
        

        df['book_id'] = df['book_id'].astype(int) ## convert book_id to integer type

        ## for debugging purposes
        # print(f'\nafter cleaning {file_path}, length: {len(df)}')
        # print(f'\ntype check:\n{df.dtypes}\n') ##

        ## debugging
        # print(f'\n first 5 rows of cleaned {file_path}:\n{df.head(5)}\n')

        ## apend this cleaned dataframe to the merged_df_list
        merged_df_list.append(df)

    ## done processing all files
        

    ## combine all dataframes in mergd_df into a single dataframe
    cleaned_df = pd.concat(merged_df_list, ignore_index=True)
    
    ## debugging
    # print(f'\nlength of combined cleaned data: {len(cleaned_df)}')
    # print(f'\n first 5 rows of combined cleaned data:\n{cleaned_df.head(5)}\n')

    ## check that the final dataframe has no dupliates and missing values
    # cleaned_df = cleaned_df.drop_duplicates(subset=['user_id', 'book_id'])
    # cleaned_df = cleaned_df.dropna(subset=['rating', 'book_id'])

    return cleaned_df


def remove_threshold_rows(cleaned_df, rating_threshold=10, user_threshold=10):
    """
    remove rows from the cleaned dataframe where
    a user has rated less than user_threshold number of books
    a book has been rated less than rating_threshold number of times
    
    """

    ## filter users; keep only rows/users with at least user_threshold number of ratings
    user_counts = cleaned_df['user_id'].value_counts()
    filtered_users = user_counts[user_counts >= user_threshold].index ## only users that meet threshold

    final_df = cleaned_df[cleaned_df['user_id'].isin(filtered_users)] ## filter df

    ## same thing for books; keep only rows/boooks with at least rating_threshold number of ratings
    book_counts = final_df['book_id'].value_counts()
    filtered_books = book_counts[book_counts >= rating_threshold].index

    final_df = final_df[final_df['book_id'].isin(filtered_books)]

    # ## debugging
    # print(f'\nlength of final filtered dataframe: {len(final_df)}')


    ## return the final filtered dataframe --> used to make the utility matrix
    return final_df



def create_utility_matrix(final_df):
    """
    create the utility matrix from the CLEANED FINAL data (dataframe)

    input: final_df; can change this to the path if wanted but will need to edit code in main function as well
    cleaned data file has structure: user_id, book_id, rating
    
    output: utility matrix (pandas dataframe)
    """

    ## build user-item matrix
    user_item = final_df.pivot_table(index='user_id', columns='book_id', values='rating', aggfunc='mean')

    user_item.fillna(np.nan, inplace=True) ## ensure missing values are NaN

    ## debugging
    #print(f'\nutility (user-item) matrix check: {user_item.head(5)}')

    return user_item




def main():
    ## main function to run the cleaning process; calls all necessary functions in order

    ## list of input file paths
    ## going to use subset of data for efficiency/testing
    ## some files have different data inputs so check each csv file beforehand to get an overview
    ## of the structure/inputs (if checking cleaning process)
    ## -- change the file paths as necessary
    file_paths = [
        '../user_rating_0_to_1000.csv',
        '../user_rating_1000_to_2000.csv', ## -- uncomment to include more data
        '../user_rating_2000_to_3000.csv',
        '../user_rating_3000_to_4000.csv',
        '../user_rating_4000_to_5000.csv',
        '../user_rating_5000_to_6000.csv',
        '../user_rating_6000_to_11000.csv'
    ]

    ## list of book metadata file paths
    ## uncomment or comment out files as necessary to include more or less data
    book_file_paths = [
        '../book1-100k.csv',
        '../book100k-200k.csv',
        '../book200k-300k.csv',
        '../book300k-400k.csv',
        '../book400k-500k.csv',
        '../book500k-600k.csv',
        '../book600k-700k.csv',
        '../book700k-800k.csv',
        '../book800k-900k.csv',
        '../book900k-1000k.csv',
        '../book1000k-1100k.csv',
        '../book1100k-1200k.csv',
        '../book1200k-1300k.csv',
        '../book1300k-1400k.csv',
        '../book1400k-1500k.csv',
        '../book1500k-1600k.csv',
        '../book1600k-1700k.csv',
        '../book1700k-1800k.csv',
        '../book1800k-1900k.csv',
        '../book1900k-2000k.csv',
        '../book2000k-3000k.csv',
        '../book3000k-4000k.csv',
        '../book4000k-5000k.csv'
    ]
    print("Current working directory:", Path.cwd())
    print("Script location:", Path(__file__).parent)
    # for path in book_file_paths:
    #     full_path = Path(path).resolve()
    #     print(f"Looking for: {full_path}")
    #     print(f"  Exists? {full_path.exists()}")
    # print("\n")

    ## output file path
    ## -- change the output path as necessary
    ##output_path = '/cleaned_data_outputcleaned_user_ratings.csv'
    ## -- can change this to absolute path if needed
    output_folder = Path('Cleaned_Data_Output')
    output_folder.mkdir(parents=True, exist_ok=True)  ## create directory if it doesn't exist

    ## create a mapping dictionary for the ratings (raw data has text ratings)
    ## each category will be mapped to a numerical value (essentially representing a 1-5 scale)
    rating_scale = {
        'did not like it': 1,
        'it was ok': 2,
        'liked it': 3,
        'really liked it': 4,
        'it was amazing': 5,
        'this user doesn\'t have any rating': np.nan
    }

    ## call the functions in order to clean the data and create the utility matrix
    ## 1. clean the data, get df with structure: user_id, book_id, rating
    cleaned_df = clean_data(file_paths, book_file_paths, rating_scale)

    ## save the cleaned combined dataframe to the output path
    #cleaned_df.to_csv(output_path, index=False)


    ## 2. remove rows based on thresholds
    final_df = remove_threshold_rows(cleaned_df, 20, 20)
    ## save the final filtered dataframe to a csv file
    #final_df.to_csv(output_folder / 'final_df.csv', index=False)
    

    ## 3. create utility matrix
    utility_matrix = create_utility_matrix(final_df)
    utility_matrix.to_csv(output_folder / 'utility_matrix_full.csv') ## save utility matrix to csv file

    ## checking the utility matrix; want to look at its shape and a preview to make sure
    ## we're set to move onto the algorithms
    ## UNCOMMENT THIS TO CHECK FOR ANY ISSUES
    print(f'\n utility matrix shape: {utility_matrix.shape}') ## rows: users, columns: books
    ## first 5 rows (users) and first 5 columns (books)
    # print(f'\n first 5 rows (users) and columns (books):\n{utility_matrix.iloc[:5, :5]}')
    # print(f'total number of users: {utility_matrix.shape[0]}')
    # print(f'total number of books: {utility_matrix.shape[1]}')
    
    ## check how many entries are NaN (missing ratings)
    total_entries = utility_matrix.size
    missing_entries = utility_matrix.isna().sum().sum()
    print(f'\n total entries in utility matrix: {total_entries}')
    print(f' number of missing entries (NaN): {missing_entries}')
    print(f' percentage of missing entries: {missing_entries / total_entries * 100:.2f}%')

    ## print statement to confirm that everything ran successfully
    print("\nDATA CLEANED AND UTILITY MATRIX CREATED SUCCESSFULLY!")



if __name__ == "__main__":
    main()


## the next steps will be to implement the recommendation algorithms using this utility matrix
## (user-item) matrix; follow the read_me instructions