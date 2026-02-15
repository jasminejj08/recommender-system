# this script performs Collaborative Filtering: Item-Item

# this file will only run after the clean_data.py file is run
## as it uses the user-item (utility) matrix from that file

## this script includes:
## 1. calculating cosine similarity matrix 
## 2. predicting a rating for a user-item pair based on similar items
##      --> this is done by hiding a known rating and predictin it
##      --> comparing the predicted rating to the known rating
## 3. calculate the error metrics for the hidden predictions
## -- from 2 and 3, we can evaluate the performance of the model
## -- once the error is known, we can:
## 4. predicting ratings for all missing ratings for particular user
## 5. generating top-n recommendations for a particular user


## import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


## this project uses cosine similarity as the similarity metric
## past research has used other metrics such as probabilistic similarity (SOURCE)
def cosine_item_similarity(utility_matrix):
    """
    compute the item-item cosine similarity matrix
    this uses sklearn's cosine_similarity function for convenience; but
    manual calculation can also be done 

    input: utility_matrix (df) loadd as a csv file

    output: item-item similarity matrix (df)
    """

    ## updated to normalize 
    ## subtract mean rating of each movie from each rating before similarity calcultion
    ## transposed matrix has rows = books, columns = users
    utility_matrix_T = utility_matrix.T
    ## calculate mean rating for each book (row mean)
    row_book_means = utility_matrix_T.mean(axis=1)
    utility_matrix_b = utility_matrix_T.sub(row_book_means, axis=0)

    ## transpose the utility matrix so that books are rows, users are columns
    utility_matrix_b = utility_matrix_b.fillna(0.0)

    ## debugging
    #print(f'utility matrix transposed head:\n{utility_matrix_T.head()}')

    ## compute cosine similarity matrix using sklearn
    similarity_mat = cosine_similarity(utility_matrix_b)

    ## converting back to dataframe
    similarity_mat_df = pd.DataFrame(
        similarity_mat,
        index=utility_matrix_b.index,
        columns=utility_matrix_b.index
    )

    ## debugging
    #print(f'item-item cosine similarity matrix df head:\n{similarity_mat_df.head()}')

    return similarity_mat_df




def predict_book_ratings(utility_matrix, item_similarity_matrix, target_user_id, target_book_id, n):
    """"
    predicting the rating of a book (target item) for a user (target user)
    
    inputs:
    - utility_matrix (df)
    - item_similarity_matrix (df)
    - target_user_id (int) --> id of user predicting rating for
    - target_book_id (int) --> id of book to predict rating for
    - n (int) --> number of top similar items to use for prediction

    outputs:
    - pred_rating (float) --> predicted rating for target book by target user

    """

    ## have the similarity matrix created already; used here

    ## get the similarity scores for the TARGET book
    ## --> from the similarity matrix, get column for target book
    ## and drop the target book itself from the series (similarity of target book with itself)
    target_book_similarities = item_similarity_matrix[target_book_id].drop(labels=[target_book_id])
    
    ## books the target user has rated already (along with their ratings)
    ## --> from the utility matrix, get row for target user; drop any Nan values (not rated)
    user_rated_books = utility_matrix.loc[target_user_id].dropna()

    ## for the books the user has rated, remove the target book itself 
    ## --> THIS IS FOR EVALUATION PURPOSES; we are hiding the target book rating 
    ## if the user hasn't rated the target book, this step doesn't do anything basically
    user_rated_books = user_rated_books.drop(labels=[target_book_id], errors='ignore')
    #print(f'books rated by user {target_user_id}:\n{user_rated_books}')

    ##  filter target_book_similarities to inlcude only books the user has rated
    commmon_similarities = target_book_similarities.reindex(user_rated_books.index).dropna()
    #print(f'similarities of target book with user rated books:\n{commmon_similarities}')
    ## keep only similarities > 0; encompass all non-negative similarities for the final prediction value
    positive_similarities = commmon_similarities[commmon_similarities > 0]
    #print(f'number of positive similar items to target book: {len(positive_similarities)}')

    ## to make a prediction, need at least 1 similar item with positive similarity
    ## in a cas where there are no such items, return user's avg rating as predicted rating
    if positive_similarities.empty:
        user_avg_rating = user_rated_books.mean()
        return user_avg_rating

    ## using the top n similar books to target book that target user has rated
    top_n_similarities = positive_similarities.sort_values(ascending=False).head(n)
    #print(f'top {n} similar books to target book {target_book_id}:\n{top_n_similarities}')

    ## denominator for weighted average formula
    denominator = top_n_similarities.sum()
    ## going to use a weighted average formula to estimate the rating
    if denominator > 0:
        numerator = (user_rated_books.reindex(top_n_similarities.index) * top_n_similarities).sum()
        predicted_rating = numerator / denominator
        pred_rating = float(np.clip(predicted_rating, 1, 5))
    else:
        return user_rated_books.mean()
    
    
    ## the final predicted rating is returned (only one value)
    return pred_rating



## this is the main function called;
def error_calculation(utility_matrix, train_n):
    """
    this function calculates error metric RMSE over 100 predictions

    inputs:
    - utility_matrix (df)
    - item_similarity_matrix (df)
    - train_n (int): number of random known ratings to use for evaluation

    outputs:
    - rmse (float): root mean squared error over all predictions

    """

    ## currently, the predict_rating function returns one prediction
    ## will need to call it multiple times to get multiple predictions
    ## going to use 100 random user-item pairs where the rating is known
    ## --> hide the rating, predict it, then compare to the true known rating
    ## get the rmse over all 100 predictions

    ## keep track of the predictions and true values
    predicted_values = []
    true_values = []
    
    ## list of known ratings in the utility matrix
    known_ratings = []
    ## for each user, get their rated books and ratings
    for user_id in utility_matrix.index:
        user_i_ratings = utility_matrix.loc[user_id].dropna()
        for book_id in user_i_ratings.index:
            known_ratings.append((user_id, book_id, user_i_ratings[book_id]))

    ## taking 100 of these known ratings randomly
    np.random.seed(0)
    ## Note: if there are less than 100 known ratings, adjust accordingly
    train_ratings_index = np.random.choice(len(known_ratings), train_n, replace=False)
    train_ratings = [known_ratings[i] for i in train_ratings_index]

    ## hiding all the test ratings at once
    temp_utility_matrix = utility_matrix.copy()
    for (user_id, book_id, true_rating) in train_ratings:
        temp_utility_matrix.loc[user_id, book_id] = np.nan

    ## use this temp_utility_matrix for predictions
    ## --> need to get the similarity matrix again based on the temp utility matrix
    temp_similarity_matrix = cosine_item_similarity(temp_utility_matrix)

    ## calculating predictions for each hidden rating
    for (user_id, book_id, true_rating) in train_ratings:
        pred_r = predict_book_ratings(temp_utility_matrix, temp_similarity_matrix, user_id, book_id, n=5)

        predicted_values.append(pred_r)
        true_values.append(true_rating)

    ## done with all predictions
    ## quick check to see length of predictions and true values list
    print(f'num of predictions: {len(predicted_values)}')
    print(f'num of true values: {len(true_values)}')

    ## calculating RMSE
    predicted_array = np.array(predicted_values)
    true_array = np.array(true_values)
    rmse = np.sqrt(np.mean((predicted_array - true_array) ** 2))

    return rmse



def top_n_recs(target_user_id, utility_matrix, similarity_matrix, n_books=5):
    """
    for each item in the item-item similarity matrix,
    get the top n most similar items

    inputs: 
    - targeT_user_id (int) --> id of user to generate recommendations for
    - utility_matrix (df)
    - item_similarity_matrix (df)
    - n_books (int): number of top similar items to return

    output: 
    - dict where keys are item ids and values are lists of top n similar item ids
    - list of tuples (book_id, predicted_rating) for the top n recommendations for target user
    """

    ## will call the predict_rating function for each unknown rating for the target_user

    ## first get all the unknown books for target user
    user_rated_books = utility_matrix.loc[target_user_id]
    unknown_books = user_rated_books[user_rated_books.isna()].index
    

    results = []
    for book_id in unknown_books:
        pred_rating = predict_book_ratings(utility_matrix, similarity_matrix, target_user_id, book_id, n=5)
        results.append((book_id, round(pred_rating, 4)))

    ## all unknwon books preditced for target user
    ## check how many predictions made --> debugging
    print(f'num of unknown books predicted for user: {len(results)}')

    ## return the top_n books with the highest predicted ratings
    results.sort(key=lambda x: x[1], reverse=True)
    top_n_recommendations = results[:n_books]

    return top_n_recommendations
    






def main():

    ## load the utility matrix
    ## Note: this file is generated from clean_data.py script; run that first!!
    ## make sure the path is correct; adjust if necessary

    current_path = Path(__file__).parent
    data_path = current_path / 'Cleaned_Data_Output' / 'utility_matrix_full.csv'

    try:
        utility_matrix = pd.read_csv(data_path, index_col=0)
    except FileNotFoundError:
        print("UTILITY MATRIX FILE NOT FOUND; run clean_data.py first to generate the utility matrix")
        return
    
    # print(f'first 5 rows of utility matrix:\n{utility_matrix.head()}')

    ## create the item-item similarity matrix
    item_similarity_matrix = cosine_item_similarity(utility_matrix)

    ## first training and evaluating the model to see how it peforms
    ## over 100 random known ratings --> hide 100 known ratings and predict them
    ## resulting RMSE will give an idea of model performance
    utility_matrix_saved = utility_matrix.copy()
    rmse = error_calculation(utility_matrix_saved, train_n=100)
    print(f'RMSE over 100 hidden (known) ratings: {rmse}')

    # rmse = error_calculation(utility_matrix_saved, train_n=300)
    # print(f'RMSE over 300 hidden (known) ratings: {rmse}')


    ## actually predicting ratings for all missing ratings for a particular user
    ## since we know how well our model performs from the error metrics,
    ## this will give us a good idea of how far our actual predictins are
    ## generate top-n recommendations for a particular user
    ## i am generating top-5 recommendations for user_id=270
    target_user_id = 270
    top_n_recommendations = top_n_recs(target_user_id, utility_matrix, item_similarity_matrix, n_books=5)
    print(f'\nTop 5 book recommendations (BOOK IDs) for user {target_user_id}:\n{top_n_recommendations}')


    print("\nCollaborative Filtering: Item-Item Complete")



if __name__ == "__main__":
    main()