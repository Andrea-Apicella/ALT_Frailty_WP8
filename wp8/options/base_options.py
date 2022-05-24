import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--lstm1_units", type=int, default=256, help="number of units in the first LSTM layer")
        self.parser.add_argument("--lstm2_units", type=int, default=128, help="number of units in the second LSTM layer")
        self.parser.add_argument("--dense_units", type=int, default=128, help="number of units in the Dense layer")
        self.parser.add_argument("--dropout", type=float, default=0.3, help="Dropout between the LSTM layer and the Dense layer")
        self.parser.add_argument("--train_actors", nargs="+", default=[], help="List of actor numbers from 1 to 8 to use as traing data. Default: empty list, so train_test_split will be used.")
        self.parser.add_argument("--val_actors", nargs="+", default=[], help="List of actor numbers from 1 to 8 to use as validation data. Default: empty list, so train_test_split will be used.")
        self.parser.add_argument("--train_cams", nargs="+", default=[], help="List of cameras numbers from 1 to 7 to use as traing data. Default: all cameras")
        self.parser.add_argument("--val_cams", nargs="+", default=[], help="List of cameras numbers from 1 to 7 to use as validation data. Default: all cameras")
        self.parser.add_argument("--micro_classes", action="store_true", help="Wheter to use micro classes instead of the three macro classes (ADL, fall, lie_down)")
        self.parser.add_argument("--split_ratio", type=float, default=0.7, help="Train-validation split ratio. Default: 0.7")
        self.parser.add_argument("--drop_offair", action="store_true", help="Wheter to drop the off_air frames in which the actor is repositioning between sequences. Default: True")
        self.parser.add_argument("--undersample", action="store_true", help="Wheter to perform an undersample of the dataset to obtain balanced classes. Default: False")

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        print("------------ Options -------------")
        for k, v in args.items():
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")
        return self.opt
