import torch as t
import torch.nn as nn
from torch.autograd import Variable

from trainvalidsplit import IO
from utils import BaseFunction
from collections import Counter


class DictField(object):
    def __init__(self, x_dict, y_dict):
        self.x_dict = x_dict
        self.y_dict = y_dict


class DataField(object):
    def __init__(self, x, y, x_dict=None, y_dict=None):
        self.x = x
        self.y = y
        assert len(self.x) == len(self.y)
        self.length = len(self.x)
        self.x_dict = self.make_dict(self.x) if x_dict is None else x_dict
        self.y_dict = self.make_dict(self.y) if y_dict is None else y_dict

    @staticmethod
    def make_dict(lines):
        sum_counter = BaseFunction.iter_sum([Counter(line) for line in lines])
        words = sorted(sum_counter.keys())
        return dict(zip(words, range(len(words))))

    def make_field(self):
        def make_line_tensor(char_lst, char_to_index):
            char_lst = [char_to_index[char] for char in char_lst]
            return char_lst

        self.x = {k: make_line_tensor(x_line, self.x_dict) for k, x_line in enumerate(self.x)}
        self.y = {k: make_line_tensor(y_line, self.y_dict) for k, y_line in enumerate(self.y)}
        return 0


class InputData(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def read(self):
        train_filepath = self.data_path + ".train"
        valid_filepath = self.data_path + ".valid"
        train_x_lines = IO.read(train_filepath + ".x")
        train_y_lines = IO.read(train_filepath + ".y")
        valid_x_lines = IO.read(valid_filepath + ".x")
        valid_y_lines = IO.read(valid_filepath + ".y")
        train_x_lines = [train_x_line.split() for train_x_line in train_x_lines]
        train_y_lines = [train_y_line.split() for train_y_line in train_y_lines]
        valid_x_lines = [valid_x_line.split() for valid_x_line in valid_x_lines]
        valid_y_lines = [valid_y_line.split() for valid_y_line in valid_y_lines]

        return {"train": [train_x_lines, train_y_lines], "valid": [valid_x_lines, valid_y_lines]}


class MakeEmbedding(object):
    def __init__(self, text_dict):
        self.text_dict = text_dict
        self.emb = None

    def make(self, embedding_type="onehot"):
        if embedding_type is not None:
            if embedding_type == "onehot":
                dict_size = len(self.text_dict)
                self.emb = nn.Embedding(dict_size, dict_size)
                self.emb.weight.data = t.eye(dict_size)

            elif isinstance(embedding_type, t.FloatTensor):
                self.emb = nn.Embedding(*embedding_type.size())
                self.emb.weight.data = embedding_type

        return self.emb


if __name__ == "__main__":
    data_path = "./data/names"
    dataset = InputData(data_path).read()

    train_field = DataField(*dataset["train"])
    train_field.make_field()

    dictionary = DictField(train_field.x_dict, train_field.y_dict)
    valid_field = DataField(*dataset["valid"], dictionary.x_dict, dictionary.y_dict)
    valid_field.make_field()

    dictionary.x_emb = MakeEmbedding(dictionary.x_dict).make()
    dictionary.y_emb = MakeEmbedding(dictionary.y_dict).make()

    # save dictionary, train and valid
    t.save(dictionary, open(data_path + ".dict.pt", "wb"))
    t.save(train_field, open(data_path + ".train.pt", "wb"))
    t.save(valid_field, open(data_path + ".valid.pt", "wb"))
