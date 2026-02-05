import seaborn as sns
import pandas as pd

def prepare_annotation(df, annotation):
    if annotation is None:
        return df, None

    common = df.columns.intersection(annotation.index)
    df = df[common]
    annotation = annotation.loc[common]

    return df, annotation

def build_annotation_colors(annotation_df):
    """
    annotation_df: index=Sample, columns=annotation
    return:
        col_colors: DataFrame (Sample x annotation)
        lut_dict: dict for legend
    """
    col_colors = pd.DataFrame(index=annotation_df.index)
    lut_dict = {}

    for col in annotation_df.columns:
        categories = annotation_df[col].astype(str).unique()
        palette = sns.color_palette("tab10", len(categories))
        lut = dict(zip(categories, palette))

        lut_dict[col] = lut
        col_colors[col] = annotation_df[col].astype(str).map(lut)

    return col_colors, lut_dict

