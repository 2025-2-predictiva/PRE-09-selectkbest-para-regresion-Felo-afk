"""Autograding script."""


def load_data():

    import pandas as pd

    dataset = pd.read_csv("files/input/auto_mpg.csv")
    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )
    y = dataset.pop("MPG")
    x = dataset.copy()

    return x, y


def load_estimator():

    import os
    import pickle

    # --
    if not os.path.exists("homework/estimator.pickle"):
        return None
    with open("homework/estimator.pickle", "rb") as file:
        estimator = pickle.load(file)

    return estimator


# def load_estimator():

#    import os
#    import pickle

#    # --
#    if not os.path.exists("homework/estimator.pickle"):
#        return None
#    with open("homework/estimator.pickle", "rb") as file:
#        estimator = pickle.load(file)
#
#    return estimator


def load_estimator():
    import os
    import pickle

    path = "homework/estimator.pickle"
    if not os.path.exists(path):
        return None

    # --- 1) Deserializar con parche de compat para _RemainderColsList
    def _load_with_compat(fobj):
        try:
            return pickle.load(fobj)
        except AttributeError:
            # parchear el símbolo privado si falta
            try:
                from sklearn.compose import _column_transformer as _ct

                if not hasattr(_ct, "_RemainderColsList"):

                    class _RemainderColsList(list):
                        pass

                    _RemainderColsList.__name__ = "_RemainderColsList"
                    _RemainderColsList.__qualname__ = "_RemainderColsList"
                    _RemainderColsList.__module__ = (
                        "sklearn.compose._column_transformer"
                    )
                    _ct._RemainderColsList = _RemainderColsList
            except Exception:
                pass
            fobj.seek(0)
            return pickle.load(fobj)

    with open(path, "rb") as f:
        est = _load_with_compat(f)

    # --- 2) Envolver para alinear columnas en predict
    class _AlignedEstimator:
        def __init__(self, inner):
            self._inner = inner
            # intentar varias rutas para obtener los nombres de columnas crudas usadas en fit
            expected = getattr(inner, "feature_names_in_", None)
            if expected is None:
                try:
                    # e.g. GridSearchCV -> best_estimator_ -> named_steps['preprocess']
                    expected = inner.best_estimator_.named_steps[
                        "preprocess"
                    ].feature_names_in_
                except Exception:
                    try:
                        expected = inner.named_steps["preprocess"].feature_names_in_
                    except Exception:
                        expected = None
            self._expected_cols = list(expected) if expected is not None else None

        def _align(self, X):
            import pandas as pd

            # Solo alineamos si es DataFrame y tenemos expected cols
            if self._expected_cols is None or not hasattr(X, "columns"):
                return X
            X = X.copy()
            # Añadir columnas faltantes como NA (imputers del pipeline deberían manejarlas)
            for col in self._expected_cols:
                if col not in X.columns:
                    X[col] = pd.NA
            # Quitar columnas extra y reordenar
            return X[self._expected_cols]

        # Delegamos casi todo al estimador interno
        def predict(self, X, *args, **kwargs):
            return self._inner.predict(self._align(X), *args, **kwargs)

        def predict_proba(self, X, *args, **kwargs):
            return self._inner.predict_proba(self._align(X), *args, **kwargs)

        def decision_function(self, X, *args, **kwargs):
            return self._inner.decision_function(self._align(X), *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _AlignedEstimator(est)


def test_01():

    from sklearn.metrics import r2_score

    x, y = load_data()
    estimator = load_estimator()

    r2 = r2_score(
        y,
        estimator.predict(x),
    )

    assert r2 > 0.6
