import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles




if __name__ == "__main__":
    plt.rcParams['figure.figsize'] = 7, 7  # graph dimensions
    plt.rcParams['font.size'] = 14  # graph font size

    X, y = make_circles(
        n_samples=6000, noise=0.1,
        shuffle=True, factor=.65
    )
    X = pd.DataFrame(X, columns=['feature1', 'feature2'])
    y = pd.Series(y)

    print('%d data points and %d features' % (X.shape))
    print('%d positive out of %d total' % (sum(y), len(y)))

    # Keep the original targets safe for later
    y_orig = y.copy()

    # Unlabel a certain number of data points
    hidden_size = 2700
    y.loc[
        np.random.choice(
            y[y == 1].index,
            replace=False,
            size=hidden_size
        )
    ] = 0

    # Check the new contents of the set
    print('%d positive out of %d total' % (sum(y), len(y)))

    # Plot the data set, as the models will see it
    plt.scatter(
        X[y == 0].feature1, X[y == 0].feature2,
        c='k', marker='.', linewidth=1, s=10, alpha=0.5,
        label='Unlabeled'
    )
    plt.scatter(
        X[y == 1].feature1, X[y == 1].feature2,
        c='b', marker='o', linewidth=0, s=50, alpha=0.5,
        label='Positive'
    )
    plt.legend()
    plt.title('Data set (as seen by the classifiers)')
    plt.show()





