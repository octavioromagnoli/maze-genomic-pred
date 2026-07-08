"""Funciones de EDA del dataset genómico (cómputo puro, sin plotting).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ("GY", "PH", "AD")
ENVIRONMENTS = ("Optimal", "Drought")
TRAIT_ENV_COLS = tuple(f"{t}_{e}" for e in ENVIRONMENTS for t in TARGETS)


def overview(geno: pd.DataFrame, pheno: pd.DataFrame) -> pd.DataFrame:
    """Resumen de shapes y solapamiento de GID entre las dos tablas."""
    g = set(geno["GID"].astype(str))
    p = set(pheno["GID"].astype(str))
    rows = [
        ("Geno_data", geno.shape[0], geno.shape[1] - 1, geno["GID"].nunique()),
        ("Pheno_data", pheno.shape[0], pheno.shape[1] - 2, pheno["GID"].nunique()),
    ]
    out = pd.DataFrame(rows, columns=["tabla", "filas", "columnas_datos", "GID_unicos"])
    out.attrs["gid_geno_only"] = len(g - p)
    out.attrs["gid_pheno_only"] = len(p - g)
    out.attrs["gid_both"] = len(g & p)
    return out


def genotype_qc_values(X: np.ndarray) -> dict:
    """Chequea que la matriz de SNPs sea binaria y sin faltantes."""
    Xf = np.asarray(X)
    n_missing = int(np.isnan(Xf).sum()) if Xf.dtype.kind == "f" else 0
    uniq = np.unique(Xf[~np.isnan(Xf)] if Xf.dtype.kind == "f" else Xf)
    return {
        "n_missing": n_missing,
        "frac_missing": n_missing / Xf.size,
        "valores_unicos": uniq.tolist(),
        "es_binaria_01": set(np.unique(Xf).tolist()) <= {0, 1},
    }


def phenotype_missingness(pheno: pd.DataFrame) -> pd.DataFrame:
    """Faltantes por columna de fenotipo (cantidad y %)."""
    cols = list(TRAIT_ENV_COLS)
    miss = pheno[cols].isna().sum()
    out = pd.DataFrame({"faltantes": miss, "pct": (100 * miss / len(pheno)).round(3)})
    return out


def allele_freq(X: np.ndarray) -> np.ndarray:
    """Frecuencia del alelo '1' por SNP (= media de la columna; líneas homocigotas)."""
    return np.asarray(X, dtype=np.float64).mean(axis=0)


def maf(X: np.ndarray) -> np.ndarray:
    """Minor allele frequency por SNP: min(f, 1-f)."""
    f = allele_freq(X)
    return np.minimum(f, 1.0 - f)


def snp_qc_table(X: np.ndarray, snp_names=None) -> pd.DataFrame:
    """Tabla por-SNP: frecuencia, MAF y flag de monomórfico."""
    f = allele_freq(X)
    m = np.minimum(f, 1.0 - f)
    idx = snp_names if snp_names is not None else np.arange(len(f))
    return pd.DataFrame({"freq": f, "maf": m, "monomorfico": m == 0.0}, index=idx)


def monomorphic_count(X: np.ndarray) -> int:
    """Cantidad de SNPs monomórficos (MAF == 0: toda la columna 0 o toda 1)."""
    return int((maf(X) == 0.0).sum())


def maf_filter_summary(maf_values: np.ndarray, thresholds=(0.0, 0.01, 0.05, 0.10)) -> pd.DataFrame:
    """Cuántos SNPs se descartan / se conservan para cada umbral de MAF."""
    maf_values = np.asarray(maf_values)
    total = len(maf_values)
    rows = []
    for t in thresholds:
        removed = int((maf_values < t).sum()) if t > 0 else int((maf_values == 0).sum())
        rows.append((t, removed, total - removed, round(100 * removed / total, 3)))
    return pd.DataFrame(rows, columns=["umbral_maf", "descartados", "conservados", "pct_descartado"])


def impute_major_allele(X_train: np.ndarray, X_apply: np.ndarray | None = None):
    """Imputación de SNPs faltantes por alelo mayoritario AJUSTADO EN TRAIN.

    Devuelve (X_apply_imputado, moda_por_snp). Pensada para usarse fold-only en
    modelado. Hoy es un no-op porque la matriz no tiene faltantes; queda como
    política codificada por si una reextracción introdujera NaN.
    """
    Xt = np.asarray(X_train, dtype=np.float64)
    major = (np.nanmean(Xt, axis=0) >= 0.5).astype(np.float64)
    target = Xt if X_apply is None else np.asarray(X_apply, dtype=np.float64)
    out = target.copy()
    nan_mask = np.isnan(out)
    if nan_mask.any():
        out[nan_mask] = np.take(major, np.where(nan_mask)[1])
    return out, major


def long_targets(pheno: pd.DataFrame) -> pd.DataFrame:
    """Pasa los 6 targets a formato largo (GID, Year, trait, env, value)."""
    m = pheno.melt(
        id_vars=["GID", "Year"],
        value_vars=list(TRAIT_ENV_COLS),
        var_name="trait_env",
        value_name="value",
    )
    m[["trait", "env"]] = m["trait_env"].str.split("_", n=1, expand=True)
    return m.dropna(subset=["value"]).reset_index(drop=True)


def target_describe_by_year(pheno: pd.DataFrame) -> pd.DataFrame:
    """Media/std/n por (trait, env, año), en formato largo y ordenado."""
    lt = long_targets(pheno)
    g = (
        lt.groupby(["trait", "env", "Year"])["value"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return g


def inter_trait_corr(pheno: pd.DataFrame, env: str, method: str = "pearson") -> pd.DataFrame:
    """Correlación entre los 3 traits dentro de un ambiente (justifica multitarea)."""
    cols = [f"{t}_{env}" for t in TARGETS]
    out = pheno[cols].corr(method=method)
    out.index = list(TARGETS)
    out.columns = list(TARGETS)
    return out


def across_env_corr(pheno: pd.DataFrame, method: str = "pearson") -> pd.Series:
    """Correlación del MISMO trait entre Optimal y Drought (mide GxE)."""
    return pd.Series(
        {t: pheno[[f"{t}_Optimal", f"{t}_Drought"]].corr(method=method).iloc[0, 1] for t in TARGETS},
        name=f"r_{method}_Optimal_vs_Drought",
    )


def full_target_corr(pheno: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Matriz de correlación de los 6 targets (trait x env)."""
    return pheno[list(TRAIT_ENV_COLS)].corr(method=method)


def outlier_summary(pheno: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Outliers por la regla de Tukey (IQR) para cada columna de target."""
    rows = []
    for c in TRAIT_ENV_COLS:
        s = pheno[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append((c, round(lo, 3), round(hi, 3), n_out, round(100 * n_out / len(s), 2), round(s.max(), 3)))
    return pd.DataFrame(rows, columns=["target", "cota_inf", "cota_sup", "n_outliers", "pct", "max"])


def pca_embed(X: np.ndarray, n_components: int = 10, fit_mask: np.ndarray | None = None, seed: int = 0):
    """PCA sobre los SNPs. Centra y ajusta con fit_mask (o todo si es None),
    luego proyecta TODAS las filas. Devuelve (Z, explained_variance_ratio).

    fit_mask permite ajustar solo en dev y proyectar el test (drift sin leakage).
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    fit_rows = X if fit_mask is None else X[fit_mask]
    mean = fit_rows.mean(axis=0)
    pca = PCA(n_components=n_components, random_state=seed).fit(fit_rows - mean)
    Z = pca.transform(X - mean)
    return Z, pca.explained_variance_ratio_


def year_centroids(Z: np.ndarray, years: np.ndarray, n_pcs: int = 2) -> pd.DataFrame:
    """Centroide (media) de cada año en las primeras n_pcs componentes."""
    years = np.asarray(years)
    rows = []
    for y in sorted(np.unique(years)):
        m = years == y
        rows.append([int(y), int(m.sum()), *Z[m, :n_pcs].mean(axis=0)])
    cols = ["Year", "n"] + [f"PC{i+1}" for i in range(n_pcs)]
    return pd.DataFrame(rows, columns=cols)


def drift_auc(
    X: np.ndarray,
    years: np.ndarray,
    pairs,
    n_pcs: int = 20,
    cv: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Separabilidad genética entre pares de años: AUC de CV de una logística
    sobre las primeras n_pcs PCs (PCA dentro del pipeline -> sin leakage del CV).

    AUC≈0.5 => años indistinguibles por genotipo; AUC≈1 => cohortes distintas
    (drift fuerte). Da evidencia directa para el protocolo de validación.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline

    X = np.asarray(X, dtype=np.float64)
    years = np.asarray(years)
    rows = []
    for ya, yb in pairs:
        m = (years == ya) | (years == yb)
        lab = (years[m] == yb).astype(int)
        pipe = make_pipeline(
            PCA(n_components=n_pcs, random_state=seed),
            LogisticRegression(max_iter=1000),
        )
        auc = cross_val_score(pipe, X[m], lab, cv=cv, scoring="roc_auc").mean()
        rows.append((f"{ya} vs {yb}", int(m.sum()), round(float(auc), 3)))
    return pd.DataFrame(rows, columns=["par_de_años", "n", "cv_auc"])


def exact_duplicate_groups(X: np.ndarray) -> dict:
    """Grupos de líneas con perfil de SNPs IDÉNTICO (clones exactos)."""
    X = np.ascontiguousarray(np.asarray(X))
    view = X.view(np.dtype((np.void, X.dtype.itemsize * X.shape[1])))
    _, counts = np.unique(view, return_counts=True)
    return {
        "perfiles_unicos": int(len(counts)),
        "grupos_duplicados": int((counts > 1).sum()),
        "lineas_en_grupos_dup": int(counts[counts > 1].sum()),
        "tamaño_max_grupo": int(counts.max()),
    }


def ibs_identity_matrix(X: np.ndarray) -> np.ndarray:
    """Matriz n×n de identidad IBS = fracción de SNPs coincidentes entre líneas.

    Diagonal puesta en -1 para ignorar la auto-comparación al tomar máximos.
    """
    Xf = np.asarray(X, dtype=np.float32)
    n, p = Xf.shape
    agree = Xf @ Xf.T + (1.0 - Xf) @ (1.0 - Xf.T)
    ident = agree / p
    np.fill_diagonal(ident, -1.0)
    return ident


def nearest_neighbor_identity(ident: np.ndarray, years: np.ndarray | None = None) -> pd.DataFrame:
    """Para cada línea: identidad IBS con su vecino más cercano (global) y, si se
    pasan years, con el vecino más cercano de OTRO año (relativo cross-split).
    """
    nn = ident.max(axis=1)
    out = pd.DataFrame({"nn_ibs": nn})
    if years is not None:
        years = np.asarray(years)
        cross = np.empty(len(years))
        for i in range(len(years)):
            other = years != years[i]
            cross[i] = ident[i, other].max() if other.any() else np.nan
        out["nn_ibs_cross_year"] = cross
    return out


def high_identity_pairs(ident: np.ndarray, years: np.ndarray, thresholds=(0.90, 0.95, 0.99)) -> pd.DataFrame:
    """Cantidad de pares de líneas con IBS por encima de cada umbral, separando
    pares del mismo año vs de años distintos (los cross-año cruzan el split temporal).
    """
    years = np.asarray(years)
    iu = np.triu_indices(ident.shape[0], 1)
    pid = ident[iu]
    yi, yj = years[iu[0]], years[iu[1]]
    rows = []
    for t in thresholds:
        m = pid > t
        same = int((yi[m] == yj[m]).sum())
        cross = int((yi[m] != yj[m]).sum())
        rows.append((t, int(m.sum()), same, cross))
    return pd.DataFrame(rows, columns=["umbral_ibs", "pares", "mismo_año", "cross_año"])
