"""
FeatureSchema – canonical column lists for preprocessing & modelling.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class FeatureSchema:
    """Container class listing every column by semantic type."""
    numerical: List[str] = field(default_factory=list)
    binary:    List[str] = field(default_factory=list)      # 0/1 ints
    ordinal:   List[str] = field(default_factory=list)      # ordered integers
    nominal:   List[str] = field(default_factory=list)      # unordered cats / high-card
    target:    str        = "success"

    # ───── convenience helpers ────────────────────────────────────
    @property
    def model_features(self) -> List[str]:
        """All predictors in modelling order."""
        return self.numerical + self.binary + self.ordinal + self.nominal

    def assert_in_dataframe(self, df) -> None:
        """Raise if any declared column is missing from df.columns."""
        missing = [c for c in self.model_features + [self.target]
                   if c not in df.columns]
        if missing:
            raise ValueError(f"FeatureSchema mismatch – missing cols: {missing}")


if __name__ == "__main__":
    # Test the feature schema
    schema = FeatureSchema()
    # sample features
    FeatureSchema.numerical = ['distance', 'weather_condition', 'time_of_day']
    FeatureSchema.binary = ['is_home', 'is_playoff']
    FeatureSchema.ordinal = ['weather_temperature']
    FeatureSchema.nominal = ['weather_type', 'team_name']
    FeatureSchema.target = 'success'
    
    print(schema.model_features)
    print(schema.assert_in_dataframe(pd.DataFrame()))
    print("******* FeatureSchema tests passed!")
