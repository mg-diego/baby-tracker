import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Start'] = pd.to_datetime(self.df['Start'], errors='coerce')
        if 'End' in self.df.columns:
            self.df['End'] = pd.to_datetime(self.df['End'], errors='coerce')

    def filter_by_category(self, category):
        return self.df[self.df['Type'] == category].copy()

    def filter_by_date_range(self, df, start_date, end_date):
        return df[(df["Start"] >= start_date) & (df["Start"] <= end_date)]