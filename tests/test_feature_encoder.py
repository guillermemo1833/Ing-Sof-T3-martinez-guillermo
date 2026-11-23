import unittest

import pandas as pd
import pandas.testing as pdt

from src.feature_encoder import FeatureEncoder


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing.

    It contains a categorical column, a numeric column and a text column
    that must remain untouched by the encoders.
    """
    return pd.DataFrame(
        {
            "city": ["SCL", "LPZ", "SCL", "BOG"],
            "age": [17, 34, 70, 45],
            "name": ["Alice", "Bob", "Carol", "Dan"],
        }
    )


class TestFeatureEncoder(unittest.TestCase):
    """Test suite for FeatureEncoder class."""

    def test_example_one_hot_with_assert_frame_equal(self):
        """Ejemplo de test usando pandas.testing.assert_frame_equal().

        Compara el DataFrame completo: valores, índices, nombres y tipos de
        columna. Es la forma correcta de comparar salidas de pandas.
        """
        df = pd.DataFrame({"color": ["red", "blue", "red"], "n": [1, 2, 3]})
        encoder = FeatureEncoder()

        result = encoder.one_hot(df, "color")

        expected = pd.DataFrame(
            {
                "n": [1, 2, 3],
                "color_blue": [0, 1, 0],
                "color_red": [1, 0, 1],
            }
        )
        pdt.assert_frame_equal(result, expected)

    def test_example_label_encode_with_assert_series_equal(self):
        """Ejemplo de test usando pandas.testing.assert_series_equal().

        Verifica valores, índice, nombre y dtype de la Serie devuelta.
        """
        df = pd.DataFrame({"size": ["M", "S", "L", "M"]})
        encoder = FeatureEncoder()

        result = encoder.label_encode(df, "size", mapping={"S": 0, "M": 1, "L": 2})

        expected = pd.Series([1, 0, 2, 1], name="size_code", dtype="int64")
        pdt.assert_series_equal(result, expected)

    def test_one_hot_creates_one_column_per_category(self):
        """Test que verifica que one_hot crea una columna indicadora por categoría
        y elimina la columna original.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a one_hot con la columna "city"
        - Verificar que "city" ya no está en las columnas del resultado (usar self.assertNotIn)
        - Verificar que existen las columnas "city_BOG", "city_LPZ" y "city_SCL" (usar self.assertIn)
        - Verificar que cada fila suma exactamente 1 en las columnas indicadoras
          (usar numpy.testing.assert_array_equal o comparar con una lista con self.assertEqual)
        """
        self.fail("TODO: implementar este test")

    def test_one_hot_raises_keyerror_for_unknown_column(self):
        """Test que verifica que one_hot lanza KeyError con una columna inexistente.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a one_hot con la columna "country"
        - Verificar que se lanza un KeyError (usar self.assertRaises)
        """
        self.fail("TODO: implementar este test")

    def test_label_encode_assigns_codes_in_sorted_order(self):
        """Test que verifica que label_encode, sin mapping, asigna códigos
        0, 1, 2, ... en orden alfabético de las categorías.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a label_encode con la columna "city" (sin mapping)
        - Construir la Serie esperada: BOG->0, LPZ->1, SCL->2, con name="city_code" y dtype int64
        - Comparar con pandas.testing.assert_series_equal
        """
        self.fail("TODO: implementar este test")

    def test_label_encode_raises_valueerror_for_unseen_category(self):
        """Test que verifica que label_encode lanza ValueError cuando el mapping
        entregado no cubre alguna categoría presente en la columna.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a label_encode con la columna "city" y mapping={"SCL": 0, "LPZ": 1}
          (falta "BOG")
        - Verificar que se lanza un ValueError (usar self.assertRaises)
        """
        self.fail("TODO: implementar este test")

    def test_bin_numeric_assigns_expected_labels(self):
        """Test que verifica que bin_numeric agrega la columna de intervalos correcta.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a bin_numeric con col="age", edges=[0, 18, 65, 120],
          labels=["young", "adult", "senior"]
        - Verificar que la columna "age_bin" contiene ["young", "adult", "senior", "adult"]
          (usar self.assertEqual comparando list(result["age_bin"]))
        - Verificar que las columnas originales siguen presentes (usar self.assertIn)
        """
        self.fail("TODO: implementar este test")

    def test_bin_numeric_raises_typeerror_for_non_numeric_column(self):
        """Test que verifica que bin_numeric lanza TypeError con una columna de texto.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df()
        - Llamar a bin_numeric con col="name", edges=[0, 1], labels=["x"]
        - Verificar que se lanza un TypeError (usar self.assertRaises)
        """
        self.fail("TODO: implementar este test")

    def test_bin_numeric_does_not_modify_original_dataframe(self):
        """Test que verifica que bin_numeric devuelve una copia y no altera el
        DataFrame de entrada.

        Escenario esperado:
        - Crear un DataFrame usando make_sample_df() y guardar una copia con df.copy()
        - Llamar a bin_numeric con col="age", edges=[0, 18, 65, 120], labels=["young", "adult", "senior"]
        - Verificar que el DataFrame original sigue igual a la copia
          (usar pandas.testing.assert_frame_equal)
        - Verificar que "age_bin" NO está en las columnas del original (usar self.assertNotIn)
        """
        self.fail("TODO: implementar este test")


if __name__ == "__main__":
    unittest.main()
