import pickle
import pandas as pd


def load_experiments(pickle_fn):
    """
    Load & return the pickle output of experiment.py.
    """
    with open(pickle_fn, 'rb') as f:
        experiments = pickle.load(f)
    return experiments

def tabulate_results(experiments):
    """
    Build and return a pandas DataFrame with one row for each Experiment
    performed.
    """
    rows = [e.summarise_results() for e in experiments]
    df = pd.DataFrame(rows)
    return df

def extract_constants(df):
    """
    Return a pandas Series indexed by the constant-value columns of the
    supplied DataFrame `df`, giving the constant value in each case.
    """
    constant_cols = []
    for col in df.columns:
        try:
            if len(set(df[col])) == 1:
                constant_cols.append(col)
        except TypeError:
            continue  # some column values are not hashable
    return df.iloc[0][constant_cols]

def _col_to_name(col):
    name = col.split('_')[0].replace('nj', 'NJ')
    name = name[0].upper() + name[1:]
    return name

def mean_normalised_rf_to_optimum(df, gbcol):
    """
    Return the mean normalised (i.e. in range [0,1]) Robinson-Foulds distance
    to the tree returned by ML tree search, grouped by `gbcol`.
    """
    cols = [col for col in df.columns if col.split('_')[-1] == 'rf-to-optimum']
    max_rf = 2 * (df.num_leaves - 3)
    nrf = df[cols].div(max_rf, axis=0).groupby(df[gbcol]).mean()
    return nrf.rename(mapper=_col_to_name, axis=1)

def mean_rf_to_generating_report(df, gbcol):
    """
    Return the mean RF distance between the inferred tree and the generating
    tree, grouped by `gbcol`.
    """
    cols = [col for col in df.columns if col.split('_')[-1] == 'rf-to-generating']
    return df[cols + [gbcol]].groupby(gbcol).mean().rename(mapper=_col_to_name, axis=1)

def _mean_topological_accuracy(rfs):
    """
    Return the rate at which the supplied Series (presumed to represent
    Robinson-Foulds distances) is equal to zero.
    """
    return (rfs == 0).mean()

def topological_accuracy_report(df, gbcol):
    """
    Return the mean rate of coincidence of the inferred tree and the generating
    tree, grouped by `gbcol`.
    """
    cols = [col for col in df.columns if col.split('_')[-1] == 'rf-to-generating']
    return df[cols + [gbcol]].groupby(gbcol).aggregate(_mean_topological_accuracy).rename(mapper=_col_to_name, axis=1)

def likelihood_report(df, gbcol, tolerance=0):
    """
    Return the rate at which the likelihood of the inferred tree exceeds the
    likelihood of the generating tree, grouped by `gbcol`.
    """
    cols = [col for col in df.columns if col.split('_')[-1] == 'll' and col != 'generating_tree_ll']
    comparisons = pd.DataFrame({col: df[col] >= (df['generating_tree_ll'] - tolerance) for col in cols}) * 1.
    comparisons[gbcol] = df[gbcol]
    rates = comparisons.groupby(gbcol).mean()
    return rates[cols].rename(mapper=_col_to_name, axis=1)

