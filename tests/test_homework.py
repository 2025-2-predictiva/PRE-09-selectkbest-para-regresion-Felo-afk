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

    path = "homework/estimator.pickle"
    if not os.path.exists(path):
        return None

    # 1) Intento normal
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except AttributeError:
        # 2) Reintento con un unpickler compatible que "backfillea" símbolos privados
        class _CompatUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # Suplente para sklearn.compose._column_transformer._RemainderColsList
                if (module, name) == (
                    "sklearn.compose._column_transformer",
                    "_RemainderColsList",
                ):
                    # Creamos una clase mínima con el nombre/calificador esperados
                    class _RemainderColsList(list):
                        pass

                    _RemainderColsList.__name__ = "_RemainderColsList"
                    _RemainderColsList.__qualname__ = "_RemainderColsList"
                    _RemainderColsList.__module__ = (
                        "sklearn.compose._column_transformer"
                    )
                    return _RemainderColsList

                # Delegar para todo lo demás
                return super().find_class(module, name)

        try:
            with open(path, "rb") as f:
                return _CompatUnpickler(f).load()
        except Exception as inner:
            raise AttributeError(
                "No se pudo cargar estimator.pickle sin fijar versión de scikit-learn. "
                "Amplía el unpickler de compatibilidad con más alias si faltan otros símbolos privados."
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
