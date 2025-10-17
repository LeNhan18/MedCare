class DataLoader:
    def __init__(self, file_path, batch_size=32):
        self.file_path = file_path
        self.batch_size = batch_size
        self.data = self.load_data()

    def load_data(self):
        # Load data from the specified file path
        # This is a placeholder for actual data loading logic
        import pandas as pd
        return pd.read_csv(self.file_path)

    def get_batches(self):
        # Generate batches of data
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i:i + self.batch_size]