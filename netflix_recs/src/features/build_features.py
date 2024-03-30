import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


def target_pipeline():
    return PowerTransformer(method="box-cox")


def get_pipeline() -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            ("target_scale", target_pipeline(), ["hours_viewed"]),
        ]
    )

    return preprocessing


def output_to_df(pipeline: Pipeline, data: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(data=data, columns=pipeline.get_feature_names_out())
