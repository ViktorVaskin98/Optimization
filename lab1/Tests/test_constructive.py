import unittest
import decimal
from Core.interval import Interval
from Core.constructive_numbers import (
    CNConstant, CNAdd, CNSub, CNMul, CNDiv, CNPow, CNExp, CNLog
)

class TestConsctuctiveNumbers(unittest.TestCase):

    def setUp(self):
        decimal.getcontext().prec = 28

    def test_constant_and_interval(self):
        c = CNConstant('5.5')

        # Не вычисляется до вызова evaluate
        self.assertEqual(c._cached_prec, -1)
        self.assertIsNone(c._cached_interval)

        # Вычисляем
        interval = c.evaluate(10)
        self.assertEqual(interval.low, decimal.Decimal('5.5'))
        self.assertEqual(interval.high, decimal.Decimal('5.5'))

        # Отработка кэша
        self.assertEqual(c._cached_prec, 10)
        self.assertIsNotNone(c._cached_interval)

    def test_tree_building_and_math(self):
        a = CNConstant(10)
        b = a + 5.5
        c = b * 2

        # Проверяем типы узлов
        self.assertIsInstance(b, CNAdd)
        self.assertIsInstance(c, CNMul)

        interval = c.evaluate(10)

        self.assertEqual(interval.low, decimal.Decimal('31'))
        self.assertEqual(interval.high, decimal.Decimal('31'))

    def test_rosenbrock_fragment_power(self):
        x = CNConstant(-2)
        y = x ** 2

        self.assertIsInstance(y, CNPow)
        interval = y.evaluate(10)

        self.assertEqual(interval.low, decimal.Decimal('4'))
        self.assertEqual(interval.high, decimal.Decimal('4'))

    def test_dynamic_precision_compare(self):
        a = CNConstant('1.0')
        b = CNConstant('1.00000000000000000001')

        self.assertTrue(a < b)
        self.assertFalse(a > b)
        self.assertFalse(a == b)

    def test_division_by_zero_interval(self):
        a = CNConstant(10)
        b = CNConstant(0)

        tree = a / b

        with self.assertRaises(ValueError):
            tree.evaluate(10)

    def test_log_and_exp(self):
        import math
        a = CNConstant(1)
        tree_exp = CNExp(a)

        interval = tree_exp.evaluate(20)
        e_approx = decimal.Decimal(str(math.e))

        self.assertAlmostEqual(float(interval.low), float(e_approx), places=7)

        with self.assertRaises(ValueError):
            CNLog(CNConstant(-5)).evaluate(10)

if __name__ == '__main__':
    unittest.main()