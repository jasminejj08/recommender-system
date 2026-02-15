# Content-Based Recommendation (TTF-IDF + Cosine Similarity)

## Note: only the following files have description column for metadata:
## book600k-700k.csv or higher

## recommending a target user books based on their previous content likes

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


## notice that only some of the files contain a description column for metadata
## the other only "content" like-columns would be Name and Author
## thus, will use TfidfVectorizer on Name + Author columns for all books
## ; anoter way to approach this would be to include description
def content_based_tfidf(books_df, utility_matrix, target_user_id, top_n=5):
    """
    this function does content-based reccommendation 
    
    inputs:
    - books_df --> dataframe containg book metadata 
    """

    books_df = books_df.iloc[:1000]

    content = books_df['unique_name'] + ' ' + books_df['authors']
    content = content.fillna('')
    content = content.str.lower().str.strip()

    # print(f'content:\n\n {content.head(3)}\n')
    # print(content.values[:3])
    # print(f'content shape: {content.shape}')

    content_vals = content.values

    #content_vals = content_vals[:1000]

    tfidf = TfidfVectorizer(stop_words='english', max_features=3000, max_df=0.7, ngram_range=(1,2))
    tfidf_result = tfidf.fit_transform(content_vals)
    print(f'TF-IDF matrix shape: {tfidf_result.shape}')

    ## to determine top-n recommended books fot target user; going to use only top rated book of user
    ## get the similarity between their top rated book and all other books --> similarity matrix
    ## return top-n most similar books that user has not rated
    user_rated_books = utility_matrix.loc[target_user_id].dropna()
    # print(f'\nuser {target_user_id} rated books:\n{user_rated_books}\n')
    if user_rated_books.empty:
        return 0
    
    # ## 
    sorted_rated_books = user_rated_books.sort_values(ascending=False)
    top_rated_book = sorted_rated_books.head(1)
    top_rated_book_id = int(sorted_rated_books.index[0]) ## index of top rated book by target user
    # print(f'\ntop rated book id for user {target_user_id}: {top_rated_book_id}\n')
    # print(top_rated_book)

    # print(f'books_df index:\n{books_df.index[:5]}\n')
    if top_rated_book_id not in books_df.index:
        print(f'top rated book id {top_rated_book_id} not in books metadata dataframe; cannot proceed with content-based recommendation')
        return 0
    ## debugging
    # else:
    #     return 1

    top_book_index = books_df.index.get_loc(top_rated_book_id)
    top_book_similarity = cosine_similarity(tfidf_result[top_book_index], tfidf_result).flatten()
    print(f'top_book_similarity shape: {top_book_similarity.shape}\n')

    top_rated_book_similarities = pd.Series(top_book_similarity, index=books_df.index)

    user_rated_books_2 = user_rated_books.index.astype(int)

    top_rated_book_similarities = top_rated_book_similarities.drop(user_rated_books_2, errors='ignore')

    top_n_books = top_rated_book_similarities.sort_values(ascending=False).head(top_n)    



    return top_n_books





def main():

    current_path = Path(__file__).parent
    metadata_path = current_path / 'Cleaned_Data_Output' / 'cleaned_books_metadata.csv.gz'

    try:
        books_metadata_df = pd.read_csv(metadata_path, index_col=0)
    except FileNotFoundError:
        print("FILE NOT FOUND; run clean_data.py first to generate the utility matrix")
        return
    
    utility_path = current_path / 'Cleaned_Data_Output' / 'utility_matrix_full.csv'
    try:
        utility_matrix = pd.read_csv(utility_path, index_col=0)
    except FileNotFoundError:
        print("FILE NOT FOUND; run clean_data.py first to generate the utility matrix")
        return
    
    ## do content-based recommendation using tfidf vectorizer
    target_user_id = 270
    top_n_recommendations = content_based_tfidf(books_metadata_df, utility_matrix, target_user_id, top_n=5)
    print(f'user = {target_user_id}\n top-n recommendations: {top_n_recommendations}')
    
    


if __name__ == "__main__":
    main()
