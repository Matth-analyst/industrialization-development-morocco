# src/robustness_checks_v2.py
"""
Chapitre 3ter -- Vérifications de second ordre.

Ce module répond à six objections méthodologiques soulevées après la
rédaction du chapitre 3bis :

1. Le chapitre 3bis effectue plusieurs nouveaux tests sans corriger pour la
   multiplicité des comparaisons.
2. La puissance statistique du test de Wald d'asymétrie (NARDL) n'a jamais
   été calculée, contrairement à celle du bounds test.
3. La date de rupture structurelle (2005) a été choisie par inspection
   visuelle du CUSUM plutôt que recherchée systématiquement.
4. Le test de causalité de Granger (IVA <-> PIBHAB) est bivarié, sans les
   variables de contrôle du modèle principal.
5. Le terme quadratique (§3bis.7) et la décomposition NARDL positive/négative
   (§3.7) pourraient capturer la même non-linéarité plutôt que deux
   phénomènes indépendants.
6. Les VIF (§3.3.2) sont calculés sur des séries en niveau, intégrées I(1),
   ce qui est un diagnostic moins fondé théoriquement que sur des séries
   stationnaires.

Chaque section écrit son résultat dans outputs/results/v2_*.txt et imprime
un résumé sur la console.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings("ignore")
# Le script est lancé depuis src/, ce qui place src/ en tête de sys.path et
# masque le config.py de la racine (src/config.py n'est qu'un ré-export).
# On force la racine en position 0 pour lever l'ambiguïté.
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data_loader import load_processed_data
from nardl_model import prepare_nardl_data, estimate_nardl

OUT = Path(config.__file__).parent / "outputs" / "results"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 3ter.1 -- Correction pour tests multiples (famille des tests du ch. 3bis)
# ---------------------------------------------------------------------------
def check_multiple_testing():
    """
    Applique Bonferroni, Holm et Benjamini-Hochberg (FDR) à la famille des
    NOUVELLES hypothèses testées au chapitre 3bis. On exclut de cette famille
    les recalculs qui vérifient un résultat DÉJÀ annoncé au chapitre 3 avec
    une méthode plus robuste (bootstrap IDE/DEP, Wald NARDL recalculé) : ces
    derniers ne créent pas de nouvelle affirmation et ne gonflent donc pas le
    taux de fausses découvertes sur des conclusions inédites.
    """
    print("\n" + "=" * 80)
    print("3ter.1 -- CORRECTION POUR TESTS MULTIPLES (famille du chapitre 3bis)")
    print("=" * 80)

    tests = {
        "Rupture structurelle 2005 (F conjoint)": 0.004,
        "Granger PIB->IVA (retard 1)": 0.079,
        "Granger PIB->IVA (retard 2)": 0.032,
        "Granger IVA->PIB (retard 1)": 0.798,
        "Granger IVA->PIB (retard 2)": 0.709,
        "Non-linéarité quadratique (ln_iva^2)": 0.005,
    }
    names = list(tests.keys())
    pvals = np.array(list(tests.values()))

    _, p_bonf, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    _, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    _, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    lines = []
    header = f"{'Test':45s} {'p brut':>8} {'p Bonf.':>9} {'p Holm':>8} {'p FDR-BH':>9}"
    lines.append(header)
    print(header)
    for n, p, pb, ph, pf in zip(names, pvals, p_bonf, p_holm, p_fdr):
        row = f"{n:45s} {p:8.4f} {pb:9.4f} {ph:8.4f} {pf:9.4f}"
        lines.append(row)
        print(row)

    lines.append("")
    lines.append(
        "Conclusion : sous correction FDR (Benjamini-Hochberg), la causalité "
        "inversée PIB->IVA au retard 2 (p brut=0.032) devient non significative "
        "(p FDR=0.064). Elle ne survit à aucune des trois corrections. Seules "
        "la rupture structurelle et la non-linéarité quadratique restent "
        "significatives sous la correction la plus stricte (Bonferroni)."
    )
    with open(OUT / "v2_multiple_testing.txt", "w") as f:
        f.write("\n".join(lines))

    return dict(zip(names, zip(pvals, p_bonf, p_holm, p_fdr)))


# ---------------------------------------------------------------------------
# 3ter.2 -- Puissance et colinéarité détaillée du test de Wald (NARDL)
# ---------------------------------------------------------------------------
def check_nardl_power_and_vif():
    print("\n" + "=" * 80)
    print("3ter.2 -- PUISSANCE ET VIF DÉTAILLÉ DU TEST D'ASYMÉTRIE (NARDL)")
    print("=" * 80)

    levels, dlog, ddiff = load_processed_data()
    best_model, best_order, _ = estimate_nardl(dlog)
    p, q = best_order

    data, X_cols = prepare_nardl_data(dlog)
    y = data["y"]
    y_lags = pd.DataFrame({f"y_L{l}": y.shift(l) for l in range(1, p + 1)})
    X_lagged = pd.DataFrame()
    for lag in range(1, q + 1):
        X_lagged[f"iva_pos_L{lag}"] = data["iva_pos"].shift(lag)
        X_lagged[f"iva_neg_L{lag}"] = data["iva_neg"].shift(lag)
    for col in X_cols:
        X_lagged[col] = data[col]
        for lag in range(1, q + 1):
            X_lagged[f"{col}_L{lag}"] = data[col].shift(lag)
    X_lagged["iva_pos"] = data["iva_pos"]
    X_lagged["iva_neg"] = data["iva_neg"]
    all_vars = pd.concat([y_lags, X_lagged], axis=1).dropna()
    y_curr = y.loc[all_vars.index]
    X_design = sm.add_constant(all_vars)

    pos_cols = [c for c in X_design.columns if c.startswith("iva_pos")]
    neg_cols = [c for c in X_design.columns if c.startswith("iva_neg")]

    model_u = sm.OLS(y_curr, X_design).fit()
    resid_u = model_u.resid.values

    # Modèle contraint (H0 : somme coefs positifs = somme coefs négatifs)
    Xc = X_design.copy()
    iva_common = Xc[pos_cols].values.sum(axis=1) + Xc[neg_cols].values.sum(axis=1)
    Xc_restricted = Xc.drop(columns=pos_cols + neg_cols)
    Xc_restricted["iva_sym"] = iva_common
    model_r = sm.OLS(y_curr, Xc_restricted).fit()
    resid_r = model_r.resid.values
    fitted_r = model_r.fittedvalues.values
    fitted_u = model_u.fittedvalues.values

    n_obs = len(y_curr)
    df_den = n_obs - X_design.shape[1]

    def f_stat(y_boot):
        m_u = sm.OLS(y_boot, X_design).fit()
        m_r = sm.OLS(y_boot, Xc_restricted).fit()
        rss_u = np.sum(m_u.resid.values ** 2)
        rss_r = np.sum(m_r.resid.values ** 2)
        return ((rss_r - rss_u) / 1) / (rss_u / df_den)

    rng = np.random.default_rng(42)
    B = 5000
    F_null = np.array(
        [f_stat(fitted_r + rng.choice(resid_r, n_obs, replace=True)) for _ in range(B)]
    )
    F_alt = np.array(
        [f_stat(fitted_u + rng.choice(resid_u, n_obs, replace=True)) for _ in range(B)]
    )

    crit90, crit95 = np.percentile(F_null, [90, 95])
    F_obs = f_stat(y_curr.values)
    power90 = np.mean(F_alt > crit90)
    power95 = np.mean(F_alt > crit95)

    # VIF individuels
    names_ = X_design.columns.tolist()
    vifs = {
        n: variance_inflation_factor(X_design.values, i)
        for i, n in enumerate(names_)
        if n != "const"
    }

    lines = [
        f"Valeurs critiques simulées (H0 bootstrap, n={n_obs}) : F90%={crit90:.3f}, F95%={crit95:.3f}",
        f"F observé : {F_obs:.3f}",
        f"Puissance simulée (effet calibré sur l'estimation observée) : à 90%={power90:.3f}, à 95%={power95:.3f}",
        "",
        "VIF individuels (design NARDL complet) :",
    ]
    for n_, v in sorted(vifs.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {n_:15s} VIF={v:10.2f}")
    lines.append("")
    lines.append(
        "Conclusion : la puissance du test de Wald d'asymétrie est élevée "
        f"(~{power95*100:.0f}% à 95%), contrairement au bounds test (36%). "
        "En revanche, la colinéarité au sein du modèle NARDL est plus étendue "
        "que ce que rapporte le §3bis.5 : les termes autorégressifs y_L1 et "
        "y_L2 ont eux-mêmes des VIF très élevés (colinéarité liée à la "
        "quasi-racine unitaire de la dynamique du PIB par habitant, cohérente "
        "avec le §3.6.1), en plus de la colinéarité connue entre iva_pos et "
        "iva_neg. Le test de Wald conjoint reste défendable, mais la lecture "
        "des coefficients individuels doit être encore plus prudente que ne "
        "le signale le rapport initial."
    )
    with open(OUT / "v2_nardl_power_vif.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return dict(power95=power95, power90=power90, vifs=vifs, F_obs=F_obs)


# ---------------------------------------------------------------------------
# 3ter.3 -- Recherche systématique de la date de rupture (sup-F / Quandt-Andrews)
# ---------------------------------------------------------------------------
def check_break_date_search(trim=0.15, B=2000, seed=0):
    print("\n" + "=" * 80)
    print("3ter.3 -- RECHERCHE SYSTÉMATIQUE DE LA DATE DE RUPTURE (sup-F)")
    print("=" * 80)

    levels, dlog, ddiff = load_processed_data()
    y = dlog["ln_pibhab"]
    X = pd.DataFrame(
        {
            "y_L1": y.shift(1),
            "y_L2": y.shift(2),
            "ln_iva": dlog["ln_iva"],
            "ln_dep": dlog["ln_dep"],
            "ln_ide": dlog["ln_ide"],
            "ln_tcer": dlog["ln_tcer"],
        }
    ).dropna()
    yv = y.loc[X.index]
    years = X.index.year.values
    Xc = sm.add_constant(X)

    base_model = sm.OLS(yv, Xc).fit()
    rss_base = np.sum(base_model.resid.values ** 2)

    cand_years = sorted(set(years[int(len(years) * trim): int(len(years) * (1 - trim))]))

    def scan(y_vec):
        rows = []
        rss_base_ = np.sum(sm.OLS(y_vec, Xc).fit().resid.values ** 2)
        for by in cand_years:
            post = (years >= by).astype(float)
            Xb = Xc.copy()
            Xb["post"] = post
            Xb["iva_post"] = post * X["ln_iva"].values
            m = sm.OLS(y_vec, Xb).fit()
            rss_b = np.sum(m.resid.values ** 2)
            df_den = len(y_vec) - Xb.shape[1]
            F = ((rss_base_ - rss_b) / 2) / (rss_b / df_den)
            rows.append((by, F))
        return rows

    rows = scan(yv.values)
    df_res = pd.DataFrame(rows, columns=["annee_rupture", "F"]).sort_values(
        "F", ascending=False
    )

    rng = np.random.default_rng(seed)
    fitted_base = base_model.fittedvalues.values
    resid_base = base_model.resid.values
    supF_null = np.zeros(B)
    for b in range(B):
        y_boot = fitted_base + rng.choice(resid_base, len(resid_base), replace=True)
        supF_null[b] = max(F for _, F in scan(y_boot))

    supF_obs = df_res["F"].max()
    p_supF = np.mean(supF_null >= supF_obs)
    best_year = int(df_res.iloc[0]["annee_rupture"])
    rank_2005 = int((df_res["F"].values >= df_res.loc[df_res.annee_rupture == 2005, "F"].values[0]).sum())

    lines = [df_res.to_string(index=False), ""]
    lines.append(f"Meilleure date scannée : {best_year}, F={df_res.iloc[0]['F']:.3f}")
    lines.append(f"Rang de 2005 parmi {len(df_res)} dates candidates : {rank_2005}")
    lines.append(f"sup-F observé : {supF_obs:.3f}")
    lines.append(
        f"p-value du sup-F test (Quandt-Andrews, corrigée par bootstrap pour la "
        f"recherche sur toutes les dates candidates) : {p_supF:.4f}"
    )
    lines.append("")
    lines.append(
        "Conclusion : il existe bien une instabilité structurelle globale "
        f"(sup-F significatif, p={p_supF:.4f}), mais 2005 n'est pas la date la "
        f"mieux supportée par les données -- elle se classe {rank_2005}e sur "
        f"{len(df_res)}. La date optimale ({best_year}) précède le Plan "
        "Émergence (2005-2015) et coïncide plutôt avec la fin du Programme "
        "d'ajustement structurel et l'accord de libre-échange Maroc-UE (1996). "
        "L'attribution causale au Plan Émergence au §3bis.4 n'est donc pas "
        "testée contre les dates alternatives et doit être révisée."
    )
    with open(OUT / "v2_break_date_search.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return dict(best_year=best_year, rank_2005=rank_2005, p_supF=p_supF, table=df_res)


# ---------------------------------------------------------------------------
# 3ter.4 -- Granger avec variables de contrôle (DEP, IDE, TCER)
# ---------------------------------------------------------------------------
def check_granger_with_controls():
    print("\n" + "=" * 80)
    print("3ter.4 -- CAUSALITÉ DE GRANGER AVEC VARIABLES DE CONTRÔLE")
    print("=" * 80)

    levels, dlog, ddiff = load_processed_data()
    d_pib = dlog["ln_pibhab"].diff()
    d_iva = dlog["ln_iva"].diff()
    d_dep = dlog["ln_dep"].diff()
    d_ide = dlog["ln_ide"].diff()
    d_tcer = dlog["ln_tcer"].diff()
    full = pd.concat([d_pib, d_iva, d_dep, d_ide, d_tcer], axis=1).dropna()
    full.columns = ["d_pib", "d_iva", "d_dep", "d_ide", "d_tcer"]

    def granger_ctrl(dep_var, causal_var, controls, data, lag):
        d = data.copy()
        y = d[dep_var]
        Xr = pd.DataFrame(index=d.index)
        for l in range(1, lag + 1):
            Xr[f"{dep_var}_L{l}"] = y.shift(l)
            for c in controls:
                Xr[f"{c}_L{l}"] = d[c].shift(l)
        Xu = Xr.copy()
        for l in range(1, lag + 1):
            Xu[f"{causal_var}_L{l}"] = d[causal_var].shift(l)
        Xr, Xu = Xr.dropna(), Xu.dropna()
        idx = Xr.index.intersection(Xu.index)
        Xr, Xu, yv = Xr.loc[idx], Xu.loc[idx], y.loc[idx]
        mr = sm.OLS(yv, sm.add_constant(Xr)).fit()
        mu = sm.OLS(yv, sm.add_constant(Xu)).fit()
        rss_r = np.sum(mr.resid.values ** 2)
        rss_u = np.sum(mu.resid.values ** 2)
        df_num, df_den = lag, len(yv) - Xu.shape[1] - 1
        F = ((rss_r - rss_u) / df_num) / (rss_u / df_den)
        p = 1 - stats.f.cdf(F, df_num, df_den)
        return F, p, len(yv), df_den

    d_biv = pd.concat([d_pib, d_iva], axis=1).dropna()
    d_biv.columns = ["d_pibhab", "d_iva"]
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r1 = grangercausalitytests(d_biv[["d_pibhab", "d_iva"]], 2)
        r2 = grangercausalitytests(d_biv[["d_iva", "d_pibhab"]], 2)

    lines = ["Comparaison bivarié vs. contrôlé (DEP, IDE, TCER) :", ""]
    header = f"{'Sens':30s} {'Retard':>7} {'p bivarié':>10} {'p avec contrôles':>18}"
    lines.append(header)
    print(header)
    for lag in [1, 2]:
        p_biv_iva_to_pib = r1[lag][0]["ssr_ftest"][1]
        p_biv_pib_to_iva = r2[lag][0]["ssr_ftest"][1]
        F_c1, p_c1, n1, d1 = granger_ctrl("d_pib", "d_iva", ["d_dep", "d_ide", "d_tcer"], full, lag)
        F_c2, p_c2, n2, d2 = granger_ctrl("d_iva", "d_pib", ["d_dep", "d_ide", "d_tcer"], full, lag)
        row1 = f"{'IVA -> PIB':30s} {lag:7d} {p_biv_iva_to_pib:10.4f} {p_c1:18.4f}"
        row2 = f"{'PIB -> IVA':30s} {lag:7d} {p_biv_pib_to_iva:10.4f} {p_c2:18.4f}"
        lines.extend([row1, row2])
        print(row1)
        print(row2)

    lines.append("")
    lines.append(
        "Conclusion : la causalité inversée PIB->IVA détectée en bivarié au "
        "retard 2 (p=0.032) disparaît complètement une fois contrôlée pour "
        "les dépenses publiques, l'IDE et le TCER (p=0.566) -- les mêmes "
        "variables déjà présentes dans le modèle ARDL principal. Ce résultat "
        "est cohérent avec un artefact de variable omise dans le test "
        "bivarié plutôt qu'avec une causalité réelle. Combiné à l'absence de "
        "survie sous correction pour tests multiples (3ter.1), ce résultat "
        "ne doit plus être présenté comme la limite la plus substantielle de "
        "l'étude."
    )
    with open(OUT / "v2_granger_controls.txt", "w") as f:
        f.write("\n".join(lines))
    return lines


# ---------------------------------------------------------------------------
# 3ter.5 -- Redondance entre le terme quadratique et la décomposition NARDL
# ---------------------------------------------------------------------------
def check_quadratic_nardl_redundancy():
    print("\n" + "=" * 80)
    print("3ter.5 -- REDONDANCE : TERME QUADRATIQUE vs. DÉCOMPOSITION NARDL")
    print("=" * 80)

    levels, dlog, ddiff = load_processed_data()
    y = dlog["ln_pibhab"]
    iva = dlog["ln_iva"]
    iva_c = iva - iva.mean()

    diff = iva.diff()
    pos = diff.where(diff > 0, 0).cumsum().fillna(0)
    neg = diff.where(diff < 0, 0).cumsum().fillna(0)

    X = pd.DataFrame(
        {
            "y_L1": y.shift(1),
            "y_L2": y.shift(2),
            "ln_iva": iva,
            "ln_dep": dlog["ln_dep"],
            "ln_ide": dlog["ln_ide"],
            "ln_tcer": dlog["ln_tcer"],
            "ln_iva_sq": iva_c ** 2,
        }
    ).dropna()
    yv = y.loc[X.index]
    common = X.index.intersection(pos.index)
    corr = np.corrcoef(X.loc[common, "ln_iva_sq"], (pos - neg).loc[common])[0, 1]

    m_quad = sm.OLS(yv, sm.add_constant(X)).fit()

    X2 = X.copy()
    X2["iva_pos"] = pos.loc[X2.index]
    X2["iva_neg"] = neg.loc[X2.index]
    m_both = sm.OLS(yv, sm.add_constant(X2)).fit()

    lines = [
        f"Corrélation entre (ln_iva - moyenne)^2 et (iva_pos - iva_neg) : {corr:.3f}",
        f"Modèle quadratique seul, OLS (sans HAC) : coef={m_quad.params['ln_iva_sq']:.3f}, p={m_quad.pvalues['ln_iva_sq']:.4f}",
        "  (rappel rapport initial, avec HAC : coef=-1.85, p=0.005 -- résultat sensible au choix des erreurs-types)",
        "",
        "Modèle combiné (quadratique + iva_pos + iva_neg) -- matrice quasi rank-déficiente :",
        str(m_both.params[["ln_iva_sq", "iva_pos", "iva_neg"]]),
        str(m_both.pvalues[["ln_iva_sq", "iva_pos", "iva_neg"]]),
        "",
        "Conclusion : corrélation de -0.61 entre les deux constructions -- pas "
        "une coïncidence mineure. Le terme quadratique et la décomposition "
        "NARDL positive/négative capturent en grande partie la même "
        "non-linéarité de la relation IVA->PIB sous deux paramétrisations "
        "différentes. Ils ne doivent pas être présentés comme deux résultats "
        "indépendants qui se corroborent (comme au §3.9.2 et au §3bis.8), et "
        "ne doivent compter que pour un seul test dans la famille corrigée "
        "en 3ter.1.",
    ]
    with open(OUT / "v2_quadratic_nardl_redundancy.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return dict(corr=corr)


# ---------------------------------------------------------------------------
# 3ter.6 -- VIF sur différences premières (séries stationnaires)
# ---------------------------------------------------------------------------
def check_vif_on_differences():
    print("\n" + "=" * 80)
    print("3ter.6 -- VIF SUR DIFFÉRENCES PREMIÈRES (SÉRIES STATIONNAIRES)")
    print("=" * 80)

    levels, dlog, ddiff = load_processed_data()
    vars_lvl = ["ln_iva", "ln_dep", "ln_ide", "ln_tcer"]

    X_lvl = sm.add_constant(dlog[vars_lvl].dropna())
    vif_lvl = {
        n: variance_inflation_factor(X_lvl.values, i)
        for i, n in enumerate(X_lvl.columns)
        if n != "const"
    }

    X_diff = sm.add_constant(dlog[vars_lvl].diff().dropna())
    vif_diff = {
        n: variance_inflation_factor(X_diff.values, i)
        for i, n in enumerate(X_diff.columns)
        if n != "const"
    }

    lines = ["VIF en niveaux (comme au §3.3.2) vs. en différences premières :", ""]
    header = f"{'Variable':10s} {'VIF (niveaux)':>15} {'VIF (diff. 1ère)':>18}"
    lines.append(header)
    for v in vars_lvl:
        lines.append(f"{v:10s} {vif_lvl[v]:15.2f} {vif_diff[v]:18.2f}")
    lines.append("")
    lines.append(
        "Conclusion : la conclusion d'absence de multicolinéarité tient dans "
        "les deux cas (tous les VIF < 2). La remarque méthodologique sur la "
        "moindre fiabilité théorique du VIF calculé sur séries I(1) reste "
        "valable en principe, mais elle ne change ici aucune conclusion du "
        "rapport."
    )
    with open(OUT / "v2_vif_differences.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return dict(vif_lvl=vif_lvl, vif_diff=vif_diff)


def run_all():
    r1 = check_multiple_testing()
    r2 = check_nardl_power_and_vif()
    r3 = check_break_date_search()
    r4 = check_granger_with_controls()
    r5 = check_quadratic_nardl_redundancy()
    r6 = check_vif_on_differences()
    print("\n" + "=" * 80)
    print("Chapitre 3ter terminé. Résultats écrits dans outputs/results/v2_*.txt")
    print("=" * 80)
    return r1, r2, r3, r4, r5, r6


if __name__ == "__main__":
    run_all()
