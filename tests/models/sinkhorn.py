import unittest

import torch
import torch.nn as nn

from models.sinkhorn import SinkhornStableNormalizer


class TestSinkhorn(unittest.TestCase):

    def setUp(self):
        self.sinkhorn = SinkhornStableNormalizer(
            epsilon=0, 
            iterations=1, 
            temp=1
        )

    def test_row_normalization(self):
        matrix = torch.tensor(
            [
                [5, 5, 5],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        row_normalized = self.sinkhorn._row_normalization(matrix)
        normalized_sum = round(torch.sum(row_normalized).item())
        
        self.assertEqual(normalized_sum, 3)

    def test_col_normalization(self):
        matrix = torch.tensor(
            [
                [5, 1, 1],
                [5, 1, 1],
                [5, 1, 1]
            ]
        )
        col_normalized = self.sinkhorn._column_normalization(matrix)
        normalized_sum = round(torch.sum(col_normalized).item())
        
        self.assertEqual(normalized_sum, 3)

    def test_forward_temp_1(self):
        matrix = torch.tensor(
            [
                [10, 10, 10],
                [2, 2, 2],
                [2, 2, 2]
            ]
        )
        output = self.sinkhorn(matrix)
        output_sum = round(torch.sum(output).item())

        self.assertEqual(output_sum, 3)

    def test_forward_temp_2(self):
        matrix = torch.tensor(
            [
                [20, 20, 20],
                [4, 4, 4],
                [4, 4, 4]
            ]
        )
        self.sinkhorn.temp = 2
        output = self.sinkhorn(matrix)
        output_sum = round(torch.sum(output).item())

        self.assertEqual(output_sum, 3)


if __name__ == "__main__":
    unittest.main()