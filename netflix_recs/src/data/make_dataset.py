# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import openpyxl as opx
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

NETFLIX_DATA = "What_We_Watched_A_Netflix_Engagement_Report_2023Jan-Jun.xlsx"


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    source_data = Path(input_filepath) / NETFLIX_DATA
    wb = opx.load_workbook(filename=source_data)
    ws = wb["Engagement"]
    df = pd.DataFrame(ws.values)

    df2 = df.copy().drop(0, axis=1)
    df2.columns = pd.Index(df2.iloc[5])
    df2 = df2.drop(range(6), axis=0)
    df2 = df2.convert_dtypes(dtype_backend="pyarrow").infer_objects()
    df2.columns = pd.Index([col.lower().replace(" ", "_").replace("?", "") for col in df2.columns])

    train_df, test_df = train_test_split(df2, random_state=42, test_size=0.1)
    for df, target in ((train_df, "train"), (test_df, "test")):
        df.to_parquet(Path(output_filepath) / f"netflix_views_raw_{target}.pq")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
