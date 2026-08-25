import unittest

from solution import EXPECTED_RECORD


class ShapeTest(unittest.TestCase):
    def test_record_shape(self):
        self.assertIsInstance(EXPECTED_RECORD, dict)
        self.assertEqual(
            set(EXPECTED_RECORD), {"index", "bucket", "status", "value", "token"}
        )


if __name__ == "__main__":
    unittest.main()
