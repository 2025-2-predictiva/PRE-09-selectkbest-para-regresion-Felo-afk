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

    # Intento normal
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except AttributeError as e:
        # Parche: backfill de símbolos privados que a veces faltan entre versiones
        try:
            from sklearn.compose import _column_transformer as _ct

            # Si el módulo existe pero no trae el símbolo, lo creamos
            if not hasattr(_ct, "_RemainderColsList"):

                class _RemainderColsList(list):
                    """Compat shim para deserializar pickles creados en otras versiones."""

                    pass

                # Asegura que el nombre calificado coincida con el del pickle
                _RemainderColsList.__name__ = "_RemainderColsList"
                _RemainderColsList.__qualname__ = "_RemainderColsList"
                _RemainderColsList.__module__ = "sklearn.compose._column_transformer"
                _ct._RemainderColsList = _RemainderColsList
        except Exception:
            # Si ni siquiera podemos importar el módulo, continuamos y dejaremos que falle abajo
            pass

        # Reintento tras parchear
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as inner:
            raise AttributeError(
                "No se pudo cargar 'estimator.pickle' por incompatibilidad de pickle "
                "entre versiones de scikit-learn. El parche para _RemainderColsList no fue suficiente."
            ) from inner


def test_01():

    from sklearn.metrics import r2_score

    x, y = load_data()
    estimator = load_estimator()

    r2 = r2_score(
        y,
        estimator.predict(x),
    )

    assert r2 > 0.6
