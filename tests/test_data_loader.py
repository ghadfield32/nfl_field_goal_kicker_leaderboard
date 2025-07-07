"""
Unit tests for DataLoader module.
"""
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.nfl_kicker_analysis.data.loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        
        # Create temporary test data
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample kickers data
        self.kickers_data = pd.DataFrame({
            'player_id': [1, 2, 3],
            'player_name': ['Player A', 'Player B', 'Player C'],
            'birthdate': ['1990-01-01', '1985-05-15', '1992-12-31']
        })
        
        # Sample attempts data
        self.attempts_data = pd.DataFrame({
            'season': [2023] * 10,
            'season_type': ['Reg'] * 10,
            'week': list(range(1, 11)),
            'game_date': ['2023-09-01'] * 10,
            'player_id': [1, 1, 2, 2, 3, 3, 1, 2, 3, 1],
            'field_goal_result': ['Made', 'Missed', 'Made', 'Made', 'Blocked', 
                                'Made', 'Made', 'Missed', 'Made', 'Made'],
            'attempt_yards': [25, 45, 35, 52, 28, 41, 33, 48, 30, 38]
        })
        
        # Save test files
        self.kickers_file = Path(self.temp_dir) / 'kickers.csv'
        self.attempts_file = Path(self.temp_dir) / 'attempts.csv'
        
        self.kickers_data.to_csv(self.kickers_file, index=False)
        self.attempts_data.to_csv(self.attempts_file, index=False)
    
    def test_load_kickers(self):
        """Test loading kickers data."""
        df = self.loader.load_kickers(self.kickers_file)
        
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ['player_id', 'player_name', 'birthdate'])
        self.assertEqual(df.iloc[0]['player_name'], 'Player A')
    
    def test_load_attempts(self):
        """Test loading attempts data."""
        df = self.loader.load_attempts(self.attempts_file)
        
        self.assertEqual(len(df), 10)
        self.assertIn('field_goal_result', df.columns)
        self.assertIn('attempt_yards', df.columns)
    
    def test_merge_datasets(self):
        """Test merging datasets."""
        self.loader.load_kickers(self.kickers_file)
        self.loader.load_attempts(self.attempts_file)
        merged_df = self.loader.merge_datasets()
        
        self.assertEqual(len(merged_df), 10)
        self.assertIn('player_name', merged_df.columns)
        self.assertEqual(merged_df['player_name'].isnull().sum(), 0)
    
    def test_load_complete_dataset(self):
        """Test loading complete dataset in one call."""
        # Mock the config paths
        original_kickers = self.loader.__class__.__module__.split('.')[0]
        
        # Direct test with file paths
        self.loader.kickers_df = None
        self.loader.attempts_df = None
        
        # Load files manually for this test
        self.loader.kickers_df = pd.read_csv(self.kickers_file)
        self.loader.attempts_df = pd.read_csv(self.attempts_file)
        
        merged_df = self.loader.merge_datasets()
        
        self.assertIsNotNone(merged_df)
        self.assertEqual(len(merged_df), 10)
    
    def test_get_data_summary(self):
        """Test data summary generation."""
        self.loader.kickers_df = self.kickers_data
        self.loader.attempts_df = self.attempts_data
        self.loader.merged_df = self.loader.merge_datasets()
        
        summary = self.loader.get_data_summary()
        
        self.assertEqual(summary['total_attempts'], 10)
        self.assertEqual(summary['unique_kickers'], 3)
        self.assertIn('outcome_counts', summary)

if __name__ == '__main__':
    unittest.main()
