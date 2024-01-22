# GRADED FUNCTION: cofi_cost_func
# UNQ_C1
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    cost = .5 * np.sum(R * np.square((X @ W.T + b) - Y))
    reg = lambda_ * .5 * (np.sum(np.square(W)) + np.sum(np.square(X)))

    return cost + reg 
