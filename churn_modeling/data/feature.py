from typing import Set

import pandas as pd
from loguru import logger
from sklearn.preprocessing import OneHotEncoder


class FeatureTransformation:

    def __init__(self, train: pd.DataFrame, categorical_columns: Set[str], remove_columns: Set[str]):
        assert set(categorical_columns).intersection(remove_columns) == set()

        self.one_hot_encoders = {}
        for col in categorical_columns:
            self.one_hot_encoders[col] = OneHotEncoder()
            self.one_hot_encoders[col].fit(train[categorical_columns])
        self.remove_columns = remove_columns

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"Columns before feature transformation: {df.columns}")

        df = df.drop(columns=self.remove_columns)

        for col, one_hot_encoder in self.one_hot_encoders.items():
            df_one_hot_encoded = one_hot_encoder.fit_transform(df[[col]])
            df = df.join(pd.DataFrame(df_one_hot_encoded.toarray(), columns=one_hot_encoder.categories_[0]))
            df = df.drop(columns=[col])

        logger.debug(f"Columns after feature transformation: {df.columns}")
        return df
