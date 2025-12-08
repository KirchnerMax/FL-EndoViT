import csv


class Log2CSV:
    def __init__(self, csv_file):
        """
        Initialize the Log2CSV class.

        Args:
            csv_file (str): Path to the CSV file where logs will be stored.
            center (str): Center name or identifier.
            cid (str): Client ID or identifier.
        """
        self.csv_file = csv_file
        self.csv_header = ['center', 'cid', 'metrics', 'epoch']

        # Create the CSV file with a header if it doesn't exist
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize the CSV file by writing the header if the file is empty."""
        try:
            with open(self.csv_file, mode='a', newline='') as file:
                file.seek(0)
                if file.tell() == 0:
                    writer = csv.writer(file)
                    writer.writerow(self.csv_header)
        except IOError as e:
            print(f"Error initializing CSV file: {e}")

    def log_metrics(self, metrics, epoch, center, cid):
        """
        Log the training and validation metrics to the CSV file.

        Args:
            log_stats (dict): Dictionary containing training metrics.
            val_stats (dict): Dictionary containing validation metrics.
            epoch (int): The epoch number.
            n_parameters (int): The number of model parameters.
        """

        # Write the metrics to the CSV file
        try:
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                for metric_name, metric_value in metrics.items():
                    row = [
                        center,  # center
                        cid,  # cid
                        f'{metric_name}: {metric_value}',  # metrics
                        epoch  # epoch
                    ]
                    writer.writerow(row)
        except IOError as e:
            print(f"Error writing to CSV file: {e}")

if __name__ == '__main__':

    # Usage example
    log_stats = {'accuracy': 0.92, 'loss': 0.15}
    val_stats = {'accuracy': 0.89, 'loss': 0.18}

    epoch = 10
    n_parameters = 10000

    train_log_stats = {**{f'train_{k}': v for k, v in log_stats.items()},
                       **{f'val_{k}': v for k, v in val_stats.items()},
                       # CHANGED 16.02.2023. f'test_{k}' -> f'val_{k}'
                       'epoch': epoch,
                       'n_parameters': n_parameters}

    csv_logger = Log2CSV(csv_file='foo.csv')

    # Log the metrics
    csv_logger.log_metrics(train_log_stats, epoch, center='center_1', cid='1')
