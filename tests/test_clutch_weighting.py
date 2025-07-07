"""
Validation tests for clutch weighting strategy.
Tests all components as specified in the technical implementation plan.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from typing import Dict, Any

try:
    from src.nfl_kicker_analysis.data.loader import DataLoader
    from src.nfl_kicker_analysis.data.feature_engineering import FeatureEngineer
    from src.nfl_kicker_analysis.data.preprocessor import DataPreprocessor
    from src.nfl_kicker_analysis.utils.metrics import EPACalculator
except ImportError:
    # For testing when package not installed
    pass


class TestClutchWeighting:
    """Test suite for clutch weighting strategy validation."""
    
    def create_sample_data(self):
        """Create sample data with clutch context for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'player_id': np.random.randint(1, 21, n_samples),
            'player_name': [f'KICKER_{i%20:02d}' for i in range(n_samples)],
            'attempt_yards': np.random.randint(20, 61, n_samples),
            'quarter': np.random.choice([1, 2, 3, 4], n_samples),
            'game_seconds_remaining': np.random.randint(0, 900, n_samples),
            'score_differential': np.random.randint(-21, 22, n_samples),
            'week': np.random.randint(1, 18, n_samples),
            'season': np.random.choice([2020, 2021, 2022], n_samples),
            'game_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'birthdate': pd.date_range('1990-01-01', periods=n_samples, freq='30D'),
            'field_goal_result': np.random.choice(['Made', 'Missed'], n_samples, p=[0.85, 0.15])
        }
        
        return pd.DataFrame(data)
    
    def test_clutch_flag_creation(self):
        """Test that clutch flag is created correctly."""
        # Create sample data
        data = self.create_sample_data()
        engineer = FeatureEngineer()
        engineered_data = engineer.create_all_features(data)
        
        # Test clutch flag exists
        assert 'is_clutch' in engineered_data.columns
        
        # Test clutch definition
        expected_clutch = (
            (engineered_data['quarter'] >= 4) &
            (engineered_data['game_seconds_remaining'] <= 120) &
            (engineered_data['score_differential'].abs() <= 3)
        ).astype(int)
        
        assert (engineered_data['is_clutch'] == expected_clutch).all()
        
        print(f"✅ Clutch flag test passed. Clutch rate: {engineered_data['is_clutch'].mean():.3f}")
    
    def test_importance_weighting(self):
        """Test importance weighting scheme."""
        data = self.create_sample_data()
        engineer = FeatureEngineer()
        engineered_data = engineer.create_all_features(data)
        
        # Test importance column exists
        assert 'importance' in engineered_data.columns
        
        # Test weighting formula: 1 + 2*clutch + 4*playoffs
        expected_importance = (
            1 + 2 * engineered_data['is_clutch'] + 4 * engineered_data['is_playoffs']
        )
        
        assert (engineered_data['importance'] == expected_importance).all()
        
        print(f"✅ Importance weighting test passed. Unique weights: {set(engineered_data['importance'].unique())}")
    
    def test_beta_binomial_shrinkage(self):
        """Test beta-binomial shrinkage implementation."""
        data = self.create_sample_data()
        engineer = FeatureEngineer()
        engineered_data = engineer.create_all_features(data)
        
        epa_calc = EPACalculator()
        clutch_ratings = epa_calc.calculate_clutch_rating_with_shrinkage(engineered_data)
        
        # Test output structure
        expected_columns = [
            'player_name', 'player_id', 'total_attempts', 'clutch_attempts',
            'clutch_made', 'raw_clutch_rate', 'clutch_rate_shrunk'
        ]
        
        for col in expected_columns:
            assert col in clutch_ratings.columns, f"Missing column: {col}"
        
        print(f"✅ Beta-binomial shrinkage test passed. {len(clutch_ratings)} kickers processed.")
    
    def run_all_tests(self):
        """Run all validation tests."""
        print("🧪 Running Clutch Weighting Validation Suite...")
        
        try:
            self.test_clutch_flag_creation()
        except Exception as e:
            print(f"❌ Clutch flag creation failed: {e}")
        
        try:
            self.test_importance_weighting()
        except Exception as e:
            print(f"❌ Importance weighting failed: {e}")
        
        try:
            self.test_beta_binomial_shrinkage()
        except Exception as e:
            print(f"❌ Beta-binomial shrinkage failed: {e}")
        
        print("\n🎉 Clutch weighting validation completed!")


def main():
    """Main test runner."""
    test_suite = TestClutchWeighting()
    test_suite.run_all_tests()


if __name__ == "__main__":
    main() 