from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs. Default: 50")
        self.parser.add_argument("--batch_size", type=int, default=60, help="Batch size in number of rows from the dataset (not in time-series). Default: 60")
        self.parser.add_argument("--seq_len", type=int, default=20, help="Lenght of one time-series. Default: 20")
        self.parser.add_argument("--stride", type=int, default=10, help="Sliding window width. Default: 10")
        self.parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate. Dafault: 0.01")
