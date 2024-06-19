import pandas as pd
import yaml


class DataProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # Extract file paths from the config file
        self.input_paths = self.config['input_paths']
        self.output_paths = self.config['output_paths']

    def read_csv(self, path: str) -> pd.DataFrame:
        """Read a CSV file and return a DataFrame."""
        return pd.read_csv(path)

    def save_csv(self, df: pd.DataFrame, path: str):
        """Save a DataFrame to a CSV file."""
        df.to_csv(path, index=False)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows where all values in the specified columns are NaN."""
        columns_to_check = list(
            set(df.columns) - set(['subject_id', 'hadm_id', 'stay_id', 'hour']))
        return df.dropna(subset=columns_to_check, how='all')

    def rolling_statistics_imputation(self, group: pd.DataFrame, window: int = 4) -> pd.DataFrame:
        """Apply rolling statistics imputation on a DataFrame group."""
        return group.fillna(group.rolling(window=window, min_periods=1).mean())

    def process_cleaning(self):
        """Process cleaning of the raw time series data."""
        raw_df = self.read_csv(self.input_paths['raw'])
        cleaned_df = self.clean_data(raw_df)
        self.save_csv(cleaned_df, self.input_paths['cleaned'])

    def process_interpolation(self):
        """Process interpolation on the cleaned time series data."""
        cleaned_df = self.read_csv(self.input_paths['cleaned'])
        interpolated_df = cleaned_df.groupby(['subject_id', 'hadm_id', 'stay_id'], group_keys=False).apply(
            lambda x: self.rolling_statistics_imputation(x)
        )
        self.save_csv(interpolated_df, self.output_paths['interpolated'])

    def process_all(self):
        """Run the full processing pipeline."""
        self.process_cleaning()
        self.process_interpolation()
        print(
            f"Interpolated data saved to {self.output_paths['interpolated']}")


def main():
    processor = DataProcessor('/home/hwxu/Projects/Dataset/PKU/KDD/Libs/config/data.yaml')
    processor.process_all()
