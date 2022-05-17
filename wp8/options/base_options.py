import argparse


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument("--lstm_units", type=int, default=32, help="number of units in the first LSTM layer")
        self.parser.add_argument("--train_actors", type=list, default=[], help="List of actor numbers from 1 to 8 to use as traing data. Default: empty list, so train_test_split will be used.")
        self.parser.add_argument("--val_actors", type=list, default=[], help="List of actor numbers from 1 to 8 to use as validation data. Default: empty list, so train_test_split will be used.")
        self.parser.add_argument("--train_cams", type=list, default=[], help="List of cameras numbers from 1 to 7 to use as traing data. Default: all cameras")
        self.parser.add_argument("--val_cams", type=list, default=[], help="List of cameras numbers from 1 to 7 to use as validation data. Default: all cameras")
        self.parser.add_argument("--split_ratio", type=float, default=0.7, help="Train-validation split ratio. Default: 70% train, 30% val")
        self.parser.add_argument("--drop_offair", type=bool, default=False, help="Wheter to drop the off_air frames in which the actor is repositioning between sequences. Default: False")
        self.parser.add_argument("--balance_classes", type=bool, default=False, help="Wheter to perform a subsample of the dataset to obtain balanced classes. Default: False")

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        args = vars(self.opt)
        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")
        return self.opt