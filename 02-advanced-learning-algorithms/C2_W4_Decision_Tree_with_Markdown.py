# UNQ_C1
# GRADED FUNCTION: compute_entropy
def compute_entropy(y):
    """
    Computes the entropy for 
    
    Args:
       y (ndarray): Numpy array indicating whether each example at a node is
           edible (`1`) or poisonous (`0`)
       
    Returns:
        entropy (float): Entropy at that node
        
    """
    if len(y) == 0:
        return 0
    
    m = len(y)
    s = np.sum(y==1)
    
    if s == 0:
        return 0
    
    if s == m:
        return 0
    
    p = s / m 
    
    return -p*np.log2(p) - ((1-p)*np.log2(1-p))

# UNQ_C2
# GRADED FUNCTION: split_dataset
def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches
    
    Args:
        X (ndarray):             Data matrix of shape(n_samples, n_features)
        node_indices (list):     List containing the active indices. I.e, the samples being considered at this step.
        feature (int):           Index of feature to split on
    
    Returns:
        left_indices (list):     Indices with feature value == 1
        right_indices (list):    Indices with feature value == 0
    """
    
    # You need to return the following variables correctly
    node_indices = np.array(node_indices)
    I = X[node_indices, feature] == 1
    negI = np.invert(I)
    left_indices = node_indices[np.where(I)].tolist()
    right_indices = node_indices[np.where(negI)].tolist()
        
    return left_indices, right_indices

# UNQ_C3
# GRADED FUNCTION: compute_information_gain
def compute_information_gain(X, y, node_indices, feature):
    
    """
    Compute the information of splitting the node on a given feature
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.
   
    Returns:
        cost (float):        Cost computed
    
    """    
    # Split dataset
    left_indices, right_indices = split_dataset(X, node_indices, feature)
    
    # Some useful variables
    X_node, y_node = X[node_indices], y[node_indices]
    X_left, y_left = X[left_indices], y[left_indices]
    X_right, y_right = X[right_indices], y[right_indices]
    
    H0, HL, HR = compute_entropy(y_node), compute_entropy(y_left), compute_entropy(y_right)
    wl = len(y_left) / len(y_node)
    wr = 1 - wl
    
    return H0 - (wl * HL + wr * HR)

# UNQ_C4
# GRADED FUNCTION: get_best_split
def get_best_split(X, y, node_indices):   
    """
    Returns the optimal feature and threshold value
    to split the node data 
    
    Args:
        X (ndarray):            Data matrix of shape(n_samples, n_features)
        y (array like):         list or ndarray with n_samples containing the target variable
        node_indices (ndarray): List containing the active indices. I.e, the samples being considered in this step.

    Returns:
        best_feature (int):     The index of the best feature to split
    """    
    
    # Some useful variables
    num_features = X.shape[1]
    
    # You need to return the following variables correctly
    info_gains = [compute_information_gain(X, y, node_indices, i) for i in range(num_features)]
    best_feature = np.argmax(np.array(info_gains))
    
    if info_gains[best_feature] == 0:
        return -1
   
    return best_feature
