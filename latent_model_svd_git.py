# this script performs: Latent Model-Based: SVD

## make sure clean_data.py has been run to generate the utility matrix
## this file will not run without it

## the method used here is matrix factorization using Singular Value Decomposition (SVD)
## to predict the ratings and recommend top n books to a target user

## note that this is the simple version of SVD
## this is NOT the version that uses gradient decsent to minimize the error

## import necessary libraries
import pandas as pd
import numpy as np
from pathlib import Path


## this is general function for a SINGLE rating prediction using SVD
## when predicting for a known target user and target book
def matrix_factorization_svd(utility_matrix, target_user_id, target_book_id, k=10):
    """
    this function implements matrix factorization using simple SVD 
    to predict a single rating for a book for a target user
    
    inputs:
    - utility_matrix --> structure: rows = users, columns = books
    - target_user_id --> the user id to predict rating for
    - target_book_id --> the book id to predict rating for
    - k --> number of singular values to consider in calculation

    outputs:
    - pred_rating --> predicted rating for target user for target book

    """

    ## utility matrix has nan for unknown ratings --> svd needs no missing values --> mentioned in report
    ## fill missing values with 0 for SVD; also transpose matrix so that
    ## rows = books (items), columns = users
    #utility_matrix_T = utility_matrix.T.fillna(0)

    ## trying the same but filling misisng values with average rating for each book
    utility_matrix_T = utility_matrix.T.apply(lambda x: x.fillna(x.mean()), axis=1)

    ratings = utility_matrix_T.values

    ## decompose utility matrix into three matrices (U, S, Vt) using SVD
    ## input data matrix = utility_matrix_T
    ## U = left singular vectors (books x k)
    ## Vt = right singular vectors (k x users)
    ## S = singular values (k x k diagonal matrix)
    U, S, Vt = np.linalg.svd(ratings, full_matrices=False)

    ## approximate the original matrix by considering only the first k singular values
    ## keeping only first k columns of U and first k rows of Vt 
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    ## reconstruct the utility matrix using the reduced matrices
    reconstructed_matrix = np.dot(np.dot(U_k, S_k), Vt_k)

    ## debugging
    #print(f'reconstructed_matrix shape: {reconstructed_matrix.shape}')

    ## change reconstructed_matrix back to dataframe
    reconstructed_df = pd.DataFrame(reconstructed_matrix, index=utility_matrix_T.index, columns=utility_matrix_T.columns)

    ## return the predicted rating for the target user and target book
    pred_rating = reconstructed_df.loc[target_book_id, target_user_id]

    pred_rating = float(np.clip(pred_rating, 1, 5))
    print(f'predicted rating for user {target_user_id} for book {target_book_id}: {pred_rating:.2f}')

    return pred_rating



## this is the main function called; it predicts ALL ratings at once
## and returns the reconstructed utility matrix with the missing predictions
## filled in
def svd_predict(utility_matrix, k=10):
    """
    this function implements matrix factorization using simple SVD 
    to predict ratings for items (books) for a target user
    
    inputs:
    - utility_matrix --> structure: rows = users, columns = books
    - k --> number of singular values to consider in calculation

    outputs:
    - reconstructed_df --> dataframe of reconstructed utility matrix with predictions

    """

    ## utility matrix has nan for unknown ratings --> svd needs no missing values --> mentioned in report
    ## fill missing values with 0 for SVD; also transpose matrix so that
    ## rows = books (items), columns = users
    #utility_matrix_T = utility_matrix.T.fillna(0)

    ## trying the same but filling misisng values with average rating for each book
    utility_matrix_T = utility_matrix.T.apply(lambda x: x.fillna(x.mean()), axis=1)

    ratings = utility_matrix_T.values

    ## decompose utility matrix into three matrices (U, S, Vt) using SVD
    ## input data matrix = utility_matrix_T
    ## U = left singular vectors (books x k)
    ## Vt = right singular vectors (k x users)
    ## S = singular values (k x k diagonal matrix)
    U, S, Vt = np.linalg.svd(ratings, full_matrices=False)

    ## approximate the original matrix by considering only the first k singular values
    ## keeping only first k columns of U and first k rows of Vt 
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]

    ## reconstruct the utility matrix using the reduced matrices
    reconstructed_matrix = np.dot(np.dot(U_k, S_k), Vt_k)

    ## debugging
    #print(f'reconstructed_matrix shape: {reconstructed_matrix.shape}')

    ## change reconstructed_matrix back to dataframe
    reconstructed_df = pd.DataFrame(reconstructed_matrix, index=utility_matrix_T.index, columns=utility_matrix_T.columns).clip(1, 5)

    ## transpose back to original format (rows = users, columns = books)
    reconstructed_df = reconstructed_df.T

    return reconstructed_df


def error_calculation(utility_matrix, train_n=100, k=10):
    """
    this function calculates rmse error metric on 100 predictions on known ratings
    sum of squared errors is also calculated to compare to rmse

    inputs:
    - utility_matrix --> structure: rows = users, columns = books
    - train_n --> number of known ratings to use for error calculation
    - k --> number of singular values to consider (latent factors)

    outputs:
    - rmse --> root mean squared error on n known ratings

    """
    ## randomly select 100 known ratings from utility matrix
    ## hide them (set to ZERO because svd required 0s)
    ## predict them and then calculate rmse

    ## fixed for correct transpose in svd_predict

    predicted_values = []
    true_values = []

    known_ratings = []

    sse = 0

    for user_id in utility_matrix.index:
        user_i_ratings = utility_matrix.loc[user_id].dropna()
        for book_id in user_i_ratings.index:
            known_ratings.append((user_id, book_id, user_i_ratings[book_id]))


    #np.random.shuffle(known_ratings)
    np.random.seed(0)
    train_ratings_index = np.random.choice(len(known_ratings), train_n, replace=False)
    train_ratings = [known_ratings[i] for i in train_ratings_index]
    # print(f'\nhad of train_ratings: {train_ratings[:5]}')

    ## create a temporary utility matrix to hide ALL known ratings that were selected for analysis
    temp_utility_matrix = utility_matrix.copy()
    for (user_id, book_id, true_rating) in train_ratings:
        temp_utility_matrix.loc[user_id, book_id] = np.nan

    ## use this temp_utility_matrix for predictions
    ## run svd prediction for each hidden rating at once
    svd_result = svd_predict(temp_utility_matrix, k=k)
    print(f'svd_result shape: {svd_result.shape}')
    #print(f'svd_result first 5 rows:\n{svd_result.head()}')



    ## calculate error
    ## fixed for transposed svd_result--transposed back in svd_predict
    ## structure of svd_result is same as utility_matrix (rows = users, columns = books)
    for (user_id, book_id, true_rating) in train_ratings:
        pred_r = np.clip(svd_result.loc[user_id, book_id], 1, 5) 

        predicted_values.append(pred_r)
        true_values.append(true_rating)
        sse += (true_rating - pred_r) ** 2


    predicted_array = np.array(predicted_values)
    true_array = np.array(true_values)
    rmse = np.sqrt(np.mean((true_array - predicted_array) ** 2)).round(2)

    rmse_from_sse = np.sqrt(sse / len(predicted_values))
    

    return rmse, rmse_from_sse


def top_n_recs(target_user_id, utility_matrix, n_books=5, k=10):
    """
    this functions generates top n book recommendations for target user
    based on svd predicted ratings

    inputs:
    - target_user_id --> the user id to predict rating for
    - utility_matrix --> structure: rows = users, columns = books
    - n_books --> number of top books to recommend

    outputs:
    top_n_books

    """

    ## indexes of the books the target user has not yet rated
    user_ratings = utility_matrix.loc[target_user_id]
    user_unrated_books = user_ratings[user_ratings.isna()].index

    # user_unrated_books = (utility_matrix.loc[target_user_id].isna())
    # unrated_books = user_unrated_books[user_unrated_books].index
    print(f'user {target_user_id} unrated books: {user_unrated_books}')

    ## use og utility matrix to predict all ratings using svd
    svd_result = svd_predict(utility_matrix, k=k)
    print(f'svd_result shape: {svd_result.shape}')
    print(f'svd_result first 5 rows:\n{svd_result.head()}')
    ## the structure of svd_result is same as utility_matrix (rows = users, columns = books)

    ## predicted ratings for target user
    user_predictions = svd_result.loc[target_user_id]

    ## from these, determine the ratings for the unrated books only
    unrated_book_predictions = user_predictions[user_unrated_books]
    # print(f'Predicted ratings for unrated books for user {target_user_id}:\n{unrated_book_predictions}')

    # print(f'output unrated book predictions: {isinstance(unrated_book_predictions, pd.Series)}')

    ## get top n books with highest predicted ratings --> top n recs
    ## order predicted ratings in descending order of rating and then get top n books
    sorted_book_predictions =  unrated_book_predictions.sort_values(ascending=False)

    top_n_books = sorted_book_predictions.head(n_books)

    return top_n_books


def main():

    ## import utility matrix
    current_path = Path(__file__).parent
    data_path = current_path / 'Cleaned_Data_Output' / 'utility_matrix_full.csv'

    try:
        utility_matrix = pd.read_csv(data_path, index_col=0)
    except FileNotFoundError:
        print("UTILITY MATRIX FILE NOT FOUND; run clean_data.py first to generate the utility matrix")
        return
    
    # print(f'utility matrix shape: {utility_matrix.shape}')
    # print(f'number of users: {utility_matrix.shape[0]}, number of books: {utility_matrix.shape[1]}')
    # print(f'unique users: {utility_matrix.index.nunique()}, unique books: {utility_matrix.columns.nunique()}')

    # print(f'first 5 rows of utility matrix:\n{utility_matrix.head()}')
    
    ## calculate error on 100 known ratings
    # rmse, rmse_from_sse = error_calculation(utility_matrix, train_n=100)
    # print(f'\nRMSE on 100 hidden (known) ratings: {rmse}')
    # print(f'RMSE calculated from SSE: {rmse_from_sse}')

    ## try different latent factors k to see how the rmse changes
    ## (checking for overfitting)
    for k in [2, 5, 10, 25]:
        rmse, rmse_from_sse = error_calculation(utility_matrix, train_n=100, k=k)
        print(f'k={k}: RMSE on 100 hidden (known) ratings: {rmse}, RMSE from SSE: {rmse_from_sse}')
    
    ## trying with larger training size
    for k in [2, 5, 10, 25]:
        rmse, rmse_from_sse = error_calculation(utility_matrix, train_n=300, k=k)
        print(f'\nk={k}: RMSE on 300 hidden (known) ratings: {rmse}, RMSE from SSE: {rmse_from_sse}')
    

    ## set target user id as 270 (as in other files)
    target_user_id = 270
    top_n_recommendations = top_n_recs(target_user_id, utility_matrix, n_books=5, k=10)
    print(f'\nTop 5 book recommendations (BOOK IDs) for user {target_user_id}:\n{top_n_recommendations}')

    print("\nLatent Model: SVD Complete")


    

if __name__ == '__main__':
    main()

