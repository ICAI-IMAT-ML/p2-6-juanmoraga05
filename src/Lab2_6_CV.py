import numpy as np

def cross_validation(model, X, y, nFolds):
    """
    Perform cross-validation on a given machine learning model to evaluate its performance.

    This function manually implements n-fold cross-validation if a specific number of folds is provided.
    If nFolds is set to -1, Leave One Out (LOO) cross-validation is performed instead, which uses each
    data point as a single test set while the rest of the data serves as the training set.

    Parameters:
    - model: scikit-learn-like estimator
        The machine learning model to be evaluated. This model must implement the .fit() and .score() methods
        similar to scikit-learn models.
    - X: array-like of shape (n_samples, n_features)
        The input features to be used for training and testing the model.
    - y: array-like of shape (n_samples,)
        The target values (class labels in classification, real numbers in regression) for the input samples.
    - nFolds: int
        The number of folds to use for cross-validation. If set to -1, LOO cross-validation is performed.

    Returns:
    - mean_score: float
        The mean score across all cross-validation folds.
    - std_score: float
        The standard deviation of the scores across all cross-validation folds, indicating the variability
        of the score across folds.

    Example:    
    --------
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import make_classification

    # Generate a synthetic dataset
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)

    # Initialize a kNN model
    model = KNeighborsClassifier(n_neighbors=5)

    # Perform 5-fold cross-validation
    mean_score, std_score = cross_validation(model, X, y, nFolds=5)

    print(f'Mean CV Score: {mean_score}, Std Deviation: {std_score}')
    """
    if nFolds == -1:
        # Implement Leave One Out CV
        nFolds = X.shape[0]

    # Calculate fold_size based on the number of folds
    fold_size = X.shape[0] // nFolds

    posiciones = np.arange(X.shape[0])

    # Initialize a list to store the accuracy values of the model for each fold
    accuracy_scores = []

    for i in range(nFolds):
        inicio_valid = i * fold_size
        fin_valid = (i + 1) * fold_size

        # Generate indices of samples for the validation set for the fold
        valid_indices = posiciones[inicio_valid:fin_valid]

        # Generate indices of samples for the training set for the fold
        train_indices = np.concatenate((posiciones[:inicio_valid], posiciones[fin_valid:]))

        # Split the dataset into training and validation
        X_train, X_valid = X[train_indices], X[valid_indices]
        y_train, y_valid = y[train_indices], y[valid_indices]

        # Train the model with the training set
        model.fit(X_train, y_train)

        # Calculate the accuracy of the model with the validation set and store it in accuracy_scores
        y_pred = model.predict(X_valid)
        accuracy = np.mean(y_pred == y_valid)
        accuracy_scores.append(accuracy)

    # Return the mean and standard deviation of the accuracy_scores
    mean_score = np.mean(accuracy_scores)
    std_score = np.std(accuracy_scores)
    return mean_score, std_score
