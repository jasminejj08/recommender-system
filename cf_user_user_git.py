# this script performs Collaborative Filtering: User-User

# this file will only run after the clean_data.py file is run
## as it uses the user-user (utility) matrix from that file

## this script is similar to the item-item cf (run that first)
## except:
## - compute user-user similarity matrix
## - find similar USERS instead of items (books)
## - predict ratings based on similar users' ratings



## import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


## this project uses cosine similarity as the similarity metric
## past research has used other metrics such as probabilistic similarity (SOURCE)
def cosine_user_similarity(utility_matrix):
    """
    compute the user-user cosine similarity matrix
    this uses sklearn's cosine_similarity function for convenience; but
    manual calculation can also be done 

    input: utility_matrix (df) loadd as a csv file

    output: user-user similarity matrix (df)
    """

    ## usser-user needs to account for user biases
    ## since user preferences are variable and people may rate differently,
    ## center data by subtracting user mean from each user's ratings before cosine similarity
    row_user_means = utility_matrix.mean(axis=1)
    utility_matrix_u = utility_matrix.sub(row_user_means, axis=0)

    ## Note: bigger rmse without normalizing the data; changed item-item to normalize as well

    ## for user-user, compute similarity between USERS;
    ## don't need to transpose utility matrix here because users are rows
    utility_matrix_u = utility_matrix_u.fillna(0.0)
    ## debugging
    #print(f'utility matrix transposed head:\n{utility_matrix_T.head()}')

    ## compute cosine similarity matrix using sklearn
    similarity_mat = cosine_similarity(utility_matrix_u)

    ## converting back to dataframe
    similarity_mat_df = pd.DataFrame(
        similarity_mat,
        index=utility_matrix_u.index,
        columns=utility_matrix_u.index
    )

    ## debugging
    #print(f'user-user cosine similarity matrix df head:\n{similarity_mat_df.head()}')

    return similarity_mat_df



def predict_book_rating(utility_matrix, user_similarity_matrix, target_user_id, target_book_id, n):
    """"
    predicting the rating of a book (target user) for a user (target user)
    
    inputs:
    - utility_matrix (df)
    - user_similarity_matrix (df) --> matrix of user-user similarities
    - target_user_id (int) --> id of user predicting rating for
    - target_book_id (int) --> id of book to predict rating for
    - n (int) --> number of top similar users to use for prediction

    outputs:
    - pred_rating (float) --> predicted rating for target book by target user

    """

    ## have the similarity matrix created already; used here

    ## get the similarity scores for the target USER
    ## --> from the similarity matrix, get column for target user
    ## and drop the target user itself from the series (similarity of target user with itself = 1)
    target_user_similarities = user_similarity_matrix[target_user_id].drop(labels=[target_user_id])
    
    ## of these users, only can use those who have rated the target book
    ## first get the ratings of the target book by all similar users from utility matrix
    ## drop.na values (users who haven't rated the target book)
    sim_user_target_book_ratings = utility_matrix[target_book_id].dropna()

    ## remove the target user from this series as well (if present)
    ## this is equivalent to hiding the target user's rating for evaluations
    sim_user_target_book_ratings = sim_user_target_book_ratings.drop(labels=[target_user_id], errors='ignore')

    ## get the similarities of the top n users who rated the target book, mapped/indexed to the ratings
    ## meaning filter the similarities to top n users along with their associated ratings
    common_similarities = target_user_similarities.reindex(sim_user_target_book_ratings.index).dropna()
    ## keep only similarities > 0; encompass only non-negative similarities for final prediction
    positive_similarities = common_similarities[common_similarities > 0]

    user_rated_books = utility_matrix.loc[target_user_id].dropna()

    ## to make a prediction, need at least 1 similar user with positive similarity
    ## in a cas where there are no such users, return user's avg rating as predicted rating
    if positive_similarities.empty:
        user_avg_rating = user_rated_books.mean()
        return user_avg_rating

    ## using the top n similar users to target user; all users must have rated target book
    top_k = positive_similarities.sort_values(ascending=False).head(n)
    #print(f'top {n} similar books to target book {target_book_id}:\n{top_n_similarities}')

    ## denominator for weighted average formula
    ## sum of similarities of top k similar users
    denominator = top_k.sum()
    ## going to use a weighted average formula to estimate the rating
    if denominator > 0:
        ## numerator is sum of (similarity * rating) for each of the top k similar users
        numerator = (sim_user_target_book_ratings.reindex(top_k.index) * top_k).sum()
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
    - utility_matrix (df) --> structure: rows = users, columns = books
    - user_similarity_matrix (df) --> structure: rows = users, columns = users
    - train_n (int): number of random known ratings to use for evaluation

    outputs:
    - rmse (float): root mean squared error over all predictions

    """

    ## currently, the predict_rating function returns one prediction
    ## will need to call it multiple times to get multiple predictions
    ## going to use 100 random user-user pairs where the rating is known
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
    print(f'train ratings sample (first 5): {train_ratings[:5]}')

    ## hide all known ratings in utility matrix --> use the similarity matrix of this 
    ## temporary utility matrix to predict ratings
    temp_utility_matrix = utility_matrix.copy()
    for (user_id, book_id, true_rating) in train_ratings:
        temp_utility_matrix.loc[user_id, book_id] = np.nan
    
    ## similarity matrix when the known ratings are hidden
    sim_matrix_hidden = cosine_user_similarity(temp_utility_matrix)

    ##
    for (user_id, book_id, true_rating) in train_ratings:

        ## predict rating
        pred_r = predict_book_rating(utility_matrix, sim_matrix_hidden, user_id, book_id, n=5)

        ## store the prediction and true rating
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



def top_n_recs(target_user_id, utility_matrix, user_similarity_matrix, n_books=5):
    """
    for the target user, generate top-n book recommendations based on predicted ratings
    --> predict ratings for all unknown books for target user first

    inputs: 
    - target_user_id (int) --> id of user to generate recommendations for
    - utility_matrix (df) --> structure: rows = users, columns = books
    - user_similarity_matrix (df) --> structure: rows = users, columns = users
    - n_books (int): number of top recommendations to generate

    output: 
    - list of tuples (book_id, predicted_rating) for the top n recommendations for target user
    """

    ## will call the predict_rating function for each unknown rating for the target_user

    ## first get all the unknown books for target user
    user_rated_books = utility_matrix.loc[target_user_id]
    unknown_books = user_rated_books[user_rated_books.isna()].index

    results = []
    for book_id in unknown_books:
        pred_rating = predict_book_rating(utility_matrix, user_similarity_matrix, target_user_id, book_id, n=5)
        results.append((book_id, round(pred_rating, 8)))

    ## all unknwon books preditced for target user
    ## check how many predictions made --> debugging
    print(f'num of unknown books predicted for user_id {target_user_id} is: {len(results)}')

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
    
    ## print(f'first 5 rows of utility matrix:\n{utility_matrix.head()}')
    ## utility_matrix is a dataframe; note this format for later use

    ## create the user-user similarity matrix
    user_similarity_matrix = cosine_user_similarity(utility_matrix)

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
    top_n_recommendations = top_n_recs(target_user_id, utility_matrix, user_similarity_matrix, n_books=5)
    print(f'Top 5 book recommendations (BOOK IDs) for user {target_user_id}:\n{top_n_recommendations}')


    print("Collaborative Filtering: user-user Complete")



if __name__ == "__main__":
    main()