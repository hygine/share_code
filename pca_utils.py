# utils/pca_utils.py

import numpy as np
import pandas as pd
from scipy.stats import chi2


def confidence_ellipse(x, y, confidence=0.95, n_points=100):
    """
    计算 PCA 2D 置信椭圆
    """
    cov = np.cov(x, y)
    mean = np.array([x.mean(), y.mean()])

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.array([np.cos(theta), np.sin(theta)])

    chi2_val = chi2.ppf(confidence, df=2)
    ellipse = eigenvecs @ np.diag(np.sqrt(eigenvals * chi2_val)) @ circle

    ellipse[0] += mean[0]
    ellipse[1] += mean[1]

    return ellipse[0], ellipse[1]


def top_loadings(pca, feature_names, pcx=1, pcy=2, top_n=20):
    """
    获取 PCA biplot 的 top loading 基因
    """
    loadings = pca.components_.T
    df = pd.DataFrame(
        loadings,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(loadings.shape[1])]
    )

    score = np.sqrt(df[f"PC{pcx}"]**2 + df[f"PC{pcy}"]**2)
    return df.loc[score.sort_values(ascending=False).head(top_n).index]
