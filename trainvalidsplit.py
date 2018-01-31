import random
import unicodedata
import string
import re
import pickle


class TrainValidSplit(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        self.length = len(self.x)
        self.valid_length = max(min(self.length // 10, 17), 1)  # 1<self.valid_length<10000
        self.train_valid_dict = {"train": {}, "valid": {}}

    def split(self):
        random.seed(0)
        self.x = random.sample(self.x, self.length)
        random.seed(0)
        self.y = random.sample(self.y, self.length)
        self.train_valid_dict["train"] = {"x": self.x[:-self.valid_length], "y": self.y[:-self.valid_length]}
        self.train_valid_dict["valid"] = {"x": self.x[-self.valid_length:], "y": self.y[-self.valid_length:]}
        print("TrainValidSplit.split: train length: %s, valid length: %s" % (
            self.length - self.valid_length, self.valid_length))
        return self.train_valid_dict

    def save(self, save_path, save_type="csv"):
        if save_type == "csv":
            train_x_path = save_path + ".train.x"
            train_y_path = save_path + ".train.y"
            IO.save("\n".join(self.train_valid_dict["train"]["x"]), train_x_path)
            IO.save("\n".join(self.train_valid_dict["train"]["y"]), train_y_path)

            valid_x_path = save_path + ".valid.x"
            valid_y_path = save_path + ".valid.y"
            IO.save("\n".join(self.train_valid_dict["valid"]["x"]), valid_x_path)
            IO.save("\n".join(self.train_valid_dict["valid"]["y"]), valid_y_path)


class Unpack(object):
    def __init__(self, sep, data_lines):
        self.sep = sep
        self.data_lines = data_lines
        self.x = []
        self.y = []

    def line_unpack(self, data_line):
        data_line = data_line[:-1]  # get rid of \n
        x, y = data_line.split(self.sep)
        return x, y

    def unpack(self):
        for data_line in self.data_lines:
            try:
                x, y = self.line_unpack(data_line)
            except ValueError as e:
                print("Unpack.unpack: from %s raise %s, %s" % (e.__class__, e.__context__, data_line))
            self.x.append(x)
            self.y.append(y)
        return self.x, self.y


class IO(object):
    def __init__(self):
        pass

    @staticmethod
    def read(read_path, read_type="csv"):
        if read_type == "csv":
            with open(read_path, "r", encoding="utf8") as f:
                return f.readlines()
        elif save_type == "pickle":
            return pickle.load(open(save_path, "rb"))
        else:
            raise NotImplementedError("other read_type is not supported yet.")

    @staticmethod
    def save(save_data, save_path, save_type="csv"):
        if save_type == "csv":
            with open(save_path, "w", encoding="utf8") as f:
                f.writelines(save_data)
        elif save_type == "pickle":
            pickle.dump(save_data, open(save_path, "wb"))
        else:
            raise NotImplementedError("other save_type is not supported yet.")


class TextClean(object):
    def __init__(self):
        self.all_letters = string.ascii_letters + " .,;'-"

    def unicode_to_ascii(self, s):
        """
        print("óŻíéõòáßäąÁêçöÉìùñãàúżłŚüèń")
        print(self.unicode_to_ascii("óŻíéõòáßäąÁêçöÉìùñãàúżłŚüèń"))

        """
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn'
                       and c in self.all_letters
                       )

    def clean(self, data_line):
        data_line = self.unicode_to_ascii(data_line)
        data_line = re.sub("[^%s]" % re.escape(self.all_letters), "", data_line)
        data_line = re.sub("[%s]" % re.escape(" .,;'-"), " ", data_line)
        return data_line.strip()

    def data_process(self, data_lines, direction="x"):
        if direction == "x":
            # del symbols, turn unicode to ascii
            data_lines = [self.clean(data_line) for data_line in data_lines]
            # data_lines = [" ".join(data_line.split()) for data_line in data_lines]

        if direction == "y":
            """
            do not make dictionary in trainvalidsplit
            """

            # data_lines_unique = set(data_lines)
            # data_to_index = dict(zip(data_lines_unique, range(len(data_lines_unique))))
            # index_to_data = dict(zip(range(len(data_lines_unique)), data_lines_unique))
            # IO.save(index_to_data, "./index_to_data.pickle", save_type="pickle")
            # print("MyTrainValidSplit.data_process: save index_to_data to ./index_to_data.pickle")
            # data_lines = ["%s" % data_to_index[data_line] for data_line in data_lines]

        return data_lines


if __name__ == '__main__':
    data_filepath = "./data/names.txt"
    data_lines = IO.read(data_filepath)
    x, y = Unpack("#", data_lines).unpack()

    text_clean = TextClean()
    x = text_clean.data_process(x, direction="x")
    y = text_clean.data_process(y, direction="y")

    train_valid_split = TrainValidSplit(x, y)
    train_valid_split.split()
    train_valid_split.save("./data/names")
