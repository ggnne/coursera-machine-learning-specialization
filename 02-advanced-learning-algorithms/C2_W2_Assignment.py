# UNQ_C1
# GRADED CELL: my_softmax
def my_softmax(z):  
    """ Softmax converts a vector of values to a probability distribution.
    Args:
      z (ndarray (N,))  : input data, N features
    Returns:
      a (ndarray (N,))  : softmax of z
    """    
    return np.exp(z) / np.sum(np.exp(z))

# UNQ_C2
# GRADED CELL: Sequential model
tf.random.set_seed(1234) # for consistent results
model = Sequential(
    [               
        tf.keras.layers.Input(shape=(400,)),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(15, activation="relu"),
        tf.keras.layers.Dense(10, activation="linear"),
    ], name = "my_model" 
)
