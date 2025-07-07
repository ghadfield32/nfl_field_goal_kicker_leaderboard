"""
Unit tests for metrics module.
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.nfl_kicker_analysis.utils.metrics import EPACalculator, ModelEvaluator

class TestEPACalculator(unittest.TestCase):
    """Test cases for EPACalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.epa_calc = EPACalculator()
        
        # Create sample data
        np.random.seed(42)
        self.sample_data = pd.DataFrame({
            'player_name': np.random.choice(['Player A', 'Player B', 'Player C'], 200),
            'player_id': np.random.choice([1, 2, 3], 200),
            'attempt_yards': np.random.randint(20, 55, 200),
            'success': np.random.choice([0, 1], 200, p=[0.15, 0.85])
        })
    
    def test_calculate_league_average_epa(self):
        """Test league average EPA calculation."""
        league_avg = self.epa_calc.calculate_league_average_epa(self.sample_data)
        
        self.assertIsInstance(league_avg, float)
        self.assertGreater(league_avg, 0)
        self.assertLess(league_avg, 3)  # Should be less than 3 points
    
    def test_calculate_empirical_success_rate(self):
        """Test empirical success rate calculation."""
        rate = self.epa_calc.calculate_empirical_success_rate(
            self.sample_data, 'Player A', 30
        )
        
        self.assertIsInstance(rate, float)
        self.assertGreaterEqual(rate, 0)
        self.assertLessEqual(rate, 1)
    
    def test_calculate_kicker_epa_plus(self):
        """Test individual kicker EPA+ calculation."""
        rating = self.epa_calc.calculate_kicker_epa_plus(
            self.sample_data, 'Player A'
        )
        
        self.assertIsInstance(rating, dict)
        self.assertIn('player_name', rating)
        self.assertIn('epa_fg_plus', rating)
        self.assertEqual(rating['player_name'], 'Player A')
    
    def test_calculate_all_kicker_ratings(self):
        """Test calculation of all kicker ratings."""
        ratings_df = self.epa_calc.calculate_all_kicker_ratings(self.sample_data)
        
        self.assertIsInstance(ratings_df, pd.DataFrame)
        self.assertGreater(len(ratings_df), 0)
        self.assertIn('epa_fg_plus', ratings_df.columns)
        self.assertIn('rank', ratings_df.columns)

class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator()
        
        # Create sample predictions
        np.random.seed(42)
        self.y_true = np.random.choice([0, 1], 100)
        self.y_pred_proba = np.random.random(100)
    
    def test_calculate_classification_metrics(self):
        """Test classification metrics calculation."""
        metrics = self.evaluator.calculate_classification_metrics(
            self.y_true, self.y_pred_proba
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('auc_roc', metrics)
        self.assertIn('brier_score', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('log_loss', metrics)
        
        # Check value ranges
        self.assertGreaterEqual(metrics['auc_roc'], 0)
        self.assertLessEqual(metrics['auc_roc'], 1)
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_compare_models(self):
        """Test model comparison functionality."""
        models_results = {
            'Model A': {'auc_roc': 0.8, 'brier_score': 0.15},
            'Model B': {'auc_roc': 0.75, 'brier_score': 0.18}
        }
        
        comparison_df = self.evaluator.compare_models(models_results)
        
        self.assertIsInstance(comparison_df, pd.DataFrame)
        self.assertEqual(len(comparison_df), 2)
        self.assertIn('auc_roc', comparison_df.columns)

if __name__ == '__main__':
    unittest.main()
