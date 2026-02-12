
import unittest
import torch
import numpy as np
from aggregator import EnhancedCAMAggregator

class TestAggregator(unittest.TestCase):
    
    def setUp(self):
        # Create dummy CAMs: 5 layers, 10x10 size
        self.cams = [torch.ones((10, 10)) * i for i in range(5)] # CAM 0=0s, CAM 1=1s...
        self.scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) # Increasing scores
        self.layer_shapes = [
            (56, 56), (56, 56), # Stage 1
            (28, 28), (28, 28), # Stage 2
            (14, 14)            # Stage 3
        ]

    def test_standard_aggregation(self):
        # Just check it runs and returns non-zero
        res = EnhancedCAMAggregator.aggregate_standard(self.cams, self.scores)
        self.assertEqual(res.shape, (10, 10))
        self.assertGreater(res.sum(), 0)

    def test_temperature_aggregation(self):
        # High Temp -> Uniform weights -> Average of CAMs (0+1+2+3+4)/5 = 2.0
        res_high = EnhancedCAMAggregator.aggregate_temperature(self.cams, self.scores, temp=100.0)
        self.assertAlmostEqual(res_high[0,0].item(), 2.0, delta=0.1)
        
        # Low Temp -> Argmax weight -> Extract CAM 4 (Score 0.5)
        res_low = EnhancedCAMAggregator.aggregate_temperature(self.cams, self.scores, temp=0.01)
        self.assertAlmostEqual(res_low[0,0].item(), 4.0, delta=0.1)

    def test_top_k_aggregation(self):
        # Top 1 -> Should be CAM 4
        res_k1 = EnhancedCAMAggregator.aggregate_top_k(self.cams, self.scores, k=1, soft=True, temp=0.01)
        self.assertAlmostEqual(res_k1[0,0].item(), 4.0, delta=0.1)
        
        # Top 2 -> CAM 3 and 4. Scores 0.4 and 0.5.
        # Softmax([0.4, 0.5]) ~ [0.47, 0.53]
        # Result ~ 0.47*3 + 0.53*4 ~ 1.41 + 2.12 = 3.53
        # Check if > 3 (CAM 3) and < 4 (CAM 4)
        res_k2_soft = EnhancedCAMAggregator.aggregate_top_k(self.cams, self.scores, k=2, soft=True, temp=1.0)
        self.assertTrue(3.0 < res_k2_soft[0,0].item() < 4.0)

    def test_stagewise_aggregation(self):
        # Shapes: [S1, S1, S2, S2, S3]
        # Scores: [0.1, 0.2, 0.3, 0.4, 0.5]
        # Stage Scores (Avg): 
        # S1: (0.1+0.2)/2 = 0.15
        # S2: (0.3+0.4)/2 = 0.35
        # S3: 0.5
        # Stage Weights should favor S3 > S2 > S1
        
        res = EnhancedCAMAggregator.aggregate_stagewise(self.cams, self.scores, self.layer_shapes)
        # Should be heavily influenced by S3 (CAM 4) and S2 (CAM 2,3)
        self.assertGreater(res[0,0].item(), 2.0) # > Average (2.0) because higher scores are later

if __name__ == '__main__':
    unittest.main()
