from pathlib import Path

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from churn_modeling import load_from_csv
from churn_modeling.data.feature import FeatureTransformation

RANDOM_SEED = 0
TEST_SIZE = 0.3


def main():
    df = load_from_csv(Path("data", "Churn_Modelling.csv"))
    logger.debug(df)

    # TODO: remove duplicate rows

    target_column = "Exited"
    y = df[target_column].to_numpy()
    rem_columns = {"RowNumber", "CustomerId", "Surname"}
    cat_columns = {"Geography", "Gender"}

    # Transform the data
    ft = FeatureTransformation(df, categorical_columns=cat_columns, remove_columns=rem_columns)
    df = ft.transform(df)
    logger.debug(df)

    X = df.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    pca = PCA(2)
    logger.debug(X_train)
    X_train_pca = pca.fit_transform(X_train)
    logger.debug(f"Explained variance: {pca.explained_variance_}")
    logger.debug(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.debug(f"Explained variance ratio sum: {sum(pca.explained_variance_ratio_)}")
    logger.debug(f"Pca components: {pca.components_}")
    logger.debug(X_train_pca)

    kmeans = KMeans(n_clusters=2, random_state=RANDOM_SEED).fit(X_train_pca)
    pred_cluster = kmeans.predict(X_train_pca)

    logger.debug(f"Accuracy train: {max(sum(pred_cluster == y_train), sum(pred_cluster != y_train)) / len(X_train)}")

    fig, ax = plt.subplots()
    pred_0 = X_train_pca[pred_cluster == 0]
    pred_1 = X_train_pca[pred_cluster == 1]
    pred_0_x, pred_0_y = list(zip(*pred_0))
    pred_1_x, pred_1_y = list(zip(*pred_1))
    ax.scatter(pred_0_x, pred_0_y, c='tab:blue',label="Pred 0")
    ax.scatter(pred_1_x, pred_1_y, c='tab:orange', label="Pred 1")

    ax.legend()
    ax.grid(True)

    plt.show()


if __name__ == '__main__':
    main()
