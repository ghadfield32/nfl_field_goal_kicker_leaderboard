"""
Data loading module for NFL kicker analysis.
Handles loading and merging of raw datasets.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings
from pandas.tseries.offsets import DateOffset

from src.nfl_kicker_analysis.config import config

class DataLoader:
    """Handles loading and initial merging of kicker datasets."""
    
    def __init__(self):
        """Initialize the data loader."""
        self.kickers_df = None
        self.attempts_df = None
        self.merged_df = None
    
    def load_kickers(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load kicker metadata.
        
        Args:
            filepath: Optional path to kickers CSV file
            
        Returns:
            DataFrame with kicker information
        """
        if filepath is None:
            filepath = config.KICKERS_FILE
            
        try:
            self.kickers_df = pd.read_csv(filepath)
            print(f"******* Loaded {len(self.kickers_df)} kickers from {filepath}")
            return self.kickers_df
        except FileNotFoundError:
            raise FileNotFoundError(f"Kickers data file not found: {filepath}")
    
    def load_attempts(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Load field goal attempts data.
        
        Args:
            filepath: Optional path to attempts CSV file
            
        Returns:
            DataFrame with field goal attempt information
        """
        if filepath is None:
            filepath = config.ATTEMPTS_FILE
            
        try:
            self.attempts_df = pd.read_csv(filepath)
            print(f"******* Loaded {len(self.attempts_df)} field goal attempts from {filepath}")
            return self.attempts_df
        except FileNotFoundError:
            raise FileNotFoundError(f"Attempts data file not found: {filepath}")
    
    def merge_datasets(self) -> pd.DataFrame:
        """
        Merge kickers and attempts datasets and drop preseason games.
        
        Returns:
            Merged DataFrame with kicker names attached to attempts, excluding preseason
        """
        if self.kickers_df is None:
            self.load_kickers()
        if self.attempts_df is None:
            self.load_attempts()
            
        # Merge on player_id
        self.merged_df = pd.merge(
            self.attempts_df,
            self.kickers_df,
            on='player_id',
            how='left'
        )
        
        # ── 3️⃣ Binary success target & drop kickers with zero makes ever ──
        self.merged_df['success'] = (self.merged_df['field_goal_result'] == 'Made').astype(int)
        makes = self.merged_df.groupby('player_id')['success'].sum()
        zero_ids = makes[makes == 0].index
        if len(zero_ids):
            print(f"🗑️  Removing {len(zero_ids)} kickers with zero makes ever")
            self.merged_df = self.merged_df.loc[~self.merged_df['player_id'].isin(zero_ids)].copy()
            
        # Drop preseason attempts if present
        if 'season_type' in self.merged_df.columns:
            preseason_mask = self.merged_df['season_type'] == 'Pre'
            num_preseason = preseason_mask.sum()
            if num_preseason > 0:
                print(f"🗑️  Filtered out {num_preseason} preseason attempts")
            # keep everything that is not Pre-season
            self.merged_df = self.merged_df.loc[~preseason_mask].copy()

        # ── 4️⃣ Filter out players with no games in the last 2 years ─────────────────
        # ensure game_date is datetime
        self.merged_df['game_date'] = pd.to_datetime(self.merged_df['game_date'])
        max_date = self.merged_df['game_date'].max()
        cutoff = max_date - DateOffset(years=2)

        # players with any game in the last 2 years
        recent = set(self.merged_df.loc[self.merged_df['game_date'] >= cutoff, 'player_id'])
        # all players
        all_ids = set(self.merged_df['player_id'])
        # players with no recent games
        inactive = all_ids - recent

        # ── 5️⃣ For those inactive players, require ≥10 games outside their last 16 ─────────────────
        keep_ids = set(recent)
        for pid in inactive:
            # get this player's games sorted newest→oldest
            pg = (self.merged_df[self.merged_df['player_id'] == pid]
                  .sort_values('game_date', ascending=False))
            # drop their 16 most‐recent games
            older = pg.iloc[16:] if len(pg) > 16 else pg.iloc[0:0]
            if len(older) >= 10:
                keep_ids.add(pid)

        # apply the combined filter
        before = len(self.merged_df)
        self.merged_df = self.merged_df[self.merged_df['player_id'].isin(keep_ids)].copy()
        after = len(self.merged_df)
        print(f"🗂️  Filtered out {before - after} attempts due to recency/experience rules")
        

        # Validate merge
        missing_kickers = self.merged_df['player_name'].isnull().sum()
        if missing_kickers > 0:
            warnings.warn(f"Found {missing_kickers} attempts with missing kicker info")

        print(f"******* Merged dataset: {self.merged_df.shape[0]} attempts, {self.merged_df.shape[1]} columns")
        return self.merged_df

    
    def load_complete_dataset(self) -> pd.DataFrame:
        """
        Load and merge complete dataset in one call.
        
        Returns:
            Complete merged DataFrame
        """
        self.load_kickers()
        self.load_attempts()
        return self.merge_datasets()
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of loaded data.
        
        Returns:
            Dictionary with data summary information
        """
        if self.merged_df is None:
            raise ValueError("No data loaded. Call load_complete_dataset() first.")
            
        summary = {
            'total_attempts': len(self.merged_df),
            'unique_kickers': self.merged_df['player_name'].nunique(),
            'unique_seasons': sorted(self.merged_df['season'].unique()),
            'season_types': self.merged_df['season_type'].unique().tolist(),
            'outcome_counts': self.merged_df['field_goal_result'].value_counts().to_dict(),
            'date_range': (
                self.merged_df['game_date'].min(),
                self.merged_df['game_date'].max()
            ),
            'distance_range': (
                self.merged_df['attempt_yards'].min(),
                self.merged_df['attempt_yards'].max()
            )
        }
        return summary

if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")
    
    loader = DataLoader()
    
    try:
        # Load complete dataset
        df = loader.load_complete_dataset()
        print("---------------head-----------------")
        print(df.head())
        print("---------------columns-----------------")
        print(df.columns)
        
        # Print summary
        summary = loader.get_data_summary()
        print("\nData Summary:")
        print(summary)
        print(f"Total attempts: {summary['total_attempts']:,}")
        print(f"Unique kickers: {summary['unique_kickers']}")
        print(f"season_types: {summary['season_types']}")
        print(f"Seasons: {summary['unique_seasons']}")
        print(f"Outcomes: {summary['outcome_counts']}")
        
        print("******* DataLoader tests passed!")
        
    except Exception as e:
        print(f"------------- Error testing DataLoader: {e}")
        print("Note: This is expected if data files are not present.")



