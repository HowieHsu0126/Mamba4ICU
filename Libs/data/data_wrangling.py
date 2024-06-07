import pandas as pd


def read_csv_files(file_paths):
    """Read CSV files and return DataFrames."""
    return [pd.read_csv(file_path) for file_path in file_paths]


def merge_dataframes_on_keys(df1, df2, keys, how='outer'):
    """Merge two DataFrames on specified keys."""
    return pd.merge(df1, df2, on=keys, how=how)


def create_subject_hour_range(df, hour_range):
    """Create a complete hour range for each subject."""
    subject_hour_df = df[['subject_id',
                          'hadm_id', 'stay_id']].drop_duplicates()
    subject_hour_df = subject_hour_df.assign(key=1).merge(
        pd.DataFrame({'hour': hour_range, 'key': 1}),
        on='key'
    ).drop('key', axis=1)
    return subject_hour_df


def merge_with_hour_range(subject_hour_df, merged_df, keys):
    """Merge the complete hour range with the original merged DataFrame."""
    return pd.merge(subject_hour_df, merged_df, on=keys, how='left')


def remove_unwanted_subjects(df, unwanted_subjects):
    """Remove rows with unwanted subject IDs."""
    return df[~df['subject_id'].isin(unwanted_subjects)].reset_index(drop=True)


def process_files():
    # Read raw data files
    vital_signs, measurements, medication, comorbidity, outcomes, cohorts = read_csv_files([
        'Input/MIMICIV/raw/vital_signs.csv',
        'Input/MIMICIV/raw/measurements.csv',
        'Input/MIMICIV/raw/medications.csv',
        'Input/MIMICIV/raw/comorbidities.csv',
        'Input/MIMICIV/raw/outcomes.csv',
        'Input/MIMICIV/raw/cohorts.csv'
    ])

    # Merge vital signs and measurements
    merged_df = merge_dataframes_on_keys(vital_signs, measurements, [
                                         'subject_id', 'hadm_id', 'stay_id', 'hour'])

    # Create complete hour range for each subject
    subject_hour_range = create_subject_hour_range(merged_df, range(1, 49))

    # Merge the complete hour range with the original merged DataFrame
    complete_df = merge_with_hour_range(subject_hour_range, merged_df, [
                                        'subject_id', 'hadm_id', 'stay_id', 'hour'])
    complete_df.to_csv('Input/MIMICIV/processed/time_series.csv', index=False)

    # Remove unwanted subjects from various data files
    unwanted_subjects = set(medication['subject_id'].unique(
    )) - set(complete_df['subject_id'].unique())
    medication = remove_unwanted_subjects(medication, unwanted_subjects)
    medication.to_csv('Input/MIMICIV/processed/medications.csv', index=False)

    unwanted_subjects = set(comorbidity['subject_id'].unique(
    )) - set(complete_df['subject_id'].unique())
    comorbidity = remove_unwanted_subjects(comorbidity, unwanted_subjects)
    comorbidity.to_csv(
        'Input/MIMICIV/processed/comorbidities.csv', index=False)

    unwanted_subjects = set(outcomes['subject_id'].unique(
    )) - set(complete_df['subject_id'].unique())
    outcomes = remove_unwanted_subjects(outcomes, unwanted_subjects)
    outcomes.to_csv('Input/MIMICIV/processed/outcomes.csv', index=False)

    unwanted_subjects = set(cohorts['subject_id'].unique(
    )) - set(complete_df['subject_id'].unique())
    demos = remove_unwanted_subjects(cohorts, unwanted_subjects).drop(
        columns=['hadm_id', 'stay_id', 'sepsis_time', 'aki_time', 'sa_aki_stage'], axis=1)
    demos.to_csv('Input/MIMICIV/processed/statics.csv', index=False)

    cohorts = remove_unwanted_subjects(cohorts, unwanted_subjects).drop(
        columns=['age', 'gender', 'race', 'admission_type'], axis=1)
    cohorts.to_csv('Input/MIMICIV/processed/cohorts.csv', index=False)


if __name__ == "__main__":
    process_files()
