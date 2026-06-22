import unittest

from ball_quant.core.handicap import handicap_result


class HandicapMappingTest(unittest.TestCase):
    def test_netherlands_minus_one(self):
        self.assertEqual(handicap_result(2, 0, -1), "home")
        self.assertEqual(handicap_result(1, 0, -1), "draw")
        self.assertEqual(handicap_result(0, 0, -1), "away")

    def test_ivory_coast_plus_one(self):
        self.assertEqual(handicap_result(0, 0, 1), "home")
        self.assertEqual(handicap_result(0, 1, 1), "draw")
        self.assertEqual(handicap_result(0, 2, 1), "away")

    def test_germany_minus_three(self):
        self.assertEqual(handicap_result(4, 0, -3), "home")
        self.assertEqual(handicap_result(3, 0, -3), "draw")
        self.assertEqual(handicap_result(2, 0, -3), "away")


if __name__ == "__main__":
    unittest.main()
