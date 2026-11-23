import os
import unittest

import numpy as np
import numpy.testing as npt

# A3 (mutación): con la variable de entorno MCD_MUTANT=1 la suite corre contra
# la versión con bug plantado. Al menos uno de sus tests debe fallar en ese modo.
if os.environ.get("MCD_MUTANT") == "1":
    from src.mutants.metrics_utils_m1 import MetricsUtils
else:
    from src.metrics_utils import MetricsUtils


class TestMetricsUtils(unittest.TestCase):
    """Test suite for MetricsUtils class."""

    def test_example_rmse_with_assert_allclose(self):
        """Ejemplo de test usando numpy.testing para un valor de punto flotante.

        numpy.testing.assert_allclose() también sirve para escalares y evita
        problemas de redondeo como 0.1 + 0.2 != 0.3.
        """
        metrics = MetricsUtils()
        y_true = [3.0, -0.5, 2.0, 7.0]
        y_pred = [2.5, 0.0, 2.0, 8.0]

        result = metrics.rmse(y_true, y_pred)

        # errores: 0.5, -0.5, 0, -1  ->  cuadrados: 0.25, 0.25, 0, 1  -> media 0.375
        expected = np.sqrt(0.375)
        npt.assert_allclose(result, expected, rtol=1e-7)

    def test_example_confusion_matrix_with_assert_array_equal(self):
        """Ejemplo de test usando numpy.testing.assert_array_equal() para arrays enteros.

        Para matrices de conteos no hay error de punto flotante, por lo que se
        usa assert_array_equal (igualdad exacta) en vez de assert_allclose.
        """
        metrics = MetricsUtils()
        y_true = ["cat", "dog", "cat", "cat", "dog"]
        y_pred = ["cat", "dog", "dog", "cat", "cat"]

        result = metrics.confusion_matrix(y_true, y_pred, labels=["cat", "dog"])

        # filas = real, columnas = predicho
        expected = np.array([[2, 1],
                             [1, 1]])
        npt.assert_array_equal(result, expected)

    def test_rmse_raises_for_length_mismatch(self):
        """Test que verifica que rmse lanza ValueError cuando y_true e y_pred
        tienen largos distintos.

        Escenario esperado:
        - Crear y_true con 3 valores e y_pred con 2 valores
        - Llamar a rmse y verificar que se lanza un ValueError (usar self.assertRaises)
        """
        self.fail("TODO: implementar este test")

    def test_mae_basic_case(self):
        """Test que verifica que mae calcula correctamente el error absoluto medio.

        Escenario esperado:
        - Elegir y_true e y_pred cuyos errores tengan signos mezclados
          (ej: y_true=[1, 2, 3], y_pred=[2, 2, 1], errores -1, 0, +2)
        - Llamar a mae y verificar el valor esperado (para el ejemplo: (1+0+2)/3 = 1.0)
          (usar numpy.testing.assert_allclose o self.assertAlmostEqual)
        """
        self.fail("TODO: implementar este test")

    def test_mae_is_zero_for_identical_inputs(self):
        """Test que verifica que mae devuelve 0 cuando la predicción es perfecta.

        Escenario esperado:
        - Usar la misma lista como y_true e y_pred
        - Verificar que mae devuelve 0.0 (usar self.assertAlmostEqual)
        """
        self.fail("TODO: implementar este test")

    def test_accuracy_basic_case(self):
        """Test que verifica que accuracy devuelve la fracción de aciertos.

        Escenario esperado:
        - Crear y_true e y_pred con 4 elementos donde coincidan exactamente 3
        - Llamar a accuracy y verificar que el resultado es 0.75 (usar self.assertAlmostEqual)
        - Verificar que el resultado está en el rango [0, 1] (usar self.assertGreaterEqual y self.assertLessEqual)
        """
        self.fail("TODO: implementar este test")

    def test_accuracy_raises_for_empty_inputs(self):
        """Test que verifica que accuracy lanza ValueError con secuencias vacías.

        Escenario esperado:
        - Llamar a accuracy con dos listas vacías
        - Verificar que se lanza un ValueError (usar self.assertRaises)
        """
        self.fail("TODO: implementar este test")

    def test_confusion_matrix_rows_are_true_labels(self):
        """Test que verifica la orientación de la matriz de confusión
        (filas = etiqueta real, columnas = etiqueta predicha).

        Escenario esperado:
        - Crear y_true e y_pred con dos clases de modo que la matriz NO sea simétrica
          (ej: y_true=[0, 0, 1], y_pred=[1, 1, 1])
        - Llamar a confusion_matrix con labels=[0, 1]
        - Verificar la matriz completa con numpy.testing.assert_array_equal
          (para el ejemplo: [[0, 2], [0, 1]])
        - Verificar que la suma de todas las celdas es igual a len(y_true) (usar self.assertEqual)
        """
        self.fail("TODO: implementar este test")

    def test_confusion_matrix_raises_for_unknown_label(self):
        """Test que verifica que confusion_matrix lanza ValueError cuando y_true o
        y_pred contienen un valor que no está en labels.

        Escenario esperado:
        - Crear y_true=["a", "b"], y_pred=["a", "c"]
        - Llamar a confusion_matrix con labels=["a", "b"]
        - Verificar que se lanza un ValueError (usar self.assertRaises)
        """
        self.fail("TODO: implementar este test")


if __name__ == "__main__":
    unittest.main()
