import unittest
from decimal import Decimal
from src.local_calculator import calculate_expression, try_calculation


class LocalCalculatorTests(unittest.TestCase):
    def test_arithmetic_precedence_and_decimals(self):
        for text,expected in [('7+8','15'),('2*(3+4)','14'),('0.1+0.2','0.3'),('12-5','7'),('6*7','42'),('(-3)*4','-12')]:
            self.assertEqual(calculate_expression(text),Decimal(expected))

    def test_natural_requests_and_label(self):
        for text,expected in [('Hi, what is 7 + 8?','15.'),('Please calculate 6 times 7.','42.'),('What is 17 plus 24?','41.'),('Which number is greater: 18 or 27?','27 is larger.'),('Which is smaller, -2 or 4?','-2 is smaller.')]:
            self.assertEqual(try_calculation(text),{'reply':expected,'source':'local_calculator'})

    def test_does_not_match_unrelated_or_compound_requests(self):
        for text in ['Explain Python 3.','I have 7 apples.','What is 2 + 2? Also explain gravity.','My password is 1234.','print(7+8)','__import__("os").system("true")','Remember 1+2 for later']:
            self.assertIsNone(try_calculation(text),text)

    def test_invalid_math_is_explicit_and_never_executed(self):
        for text in ['1/0','2**999999','2//3']:
            self.assertIn('cannot calculate',try_calculation(text)['reply'])
        with self.assertRaises(ValueError):
            calculate_expression('a+1')


if __name__=='__main__':
    unittest.main()
