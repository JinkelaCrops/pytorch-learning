import torch as t
import numpy as np
from torch.autograd import Variable
from torch.utils.data import dataloader
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime
from preprocess import DictField
from preprocess import DataField


class InputData(object):
    def __init__(self, data_path):
        self.data_path = data_path

    def read(self):
        dictionary = t.load(open(data_path + ".dict.pt", "rb"))
        train_field = t.load(open(data_path + ".train.pt", "rb"))
        valid_field = t.load(open(data_path + ".valid.pt", "rb"))
        return {"dict": dictionary, "train": train_field, "valid": valid_field}


class DataToTensor(object):
    def __init__(self, nums, emb, trunc_size=20):
        pass
    #TODO: mini batch


class DataSet(data.Dataset):
    def __init__(self, field, embedding=None):
        self.field = field
        self.embedding = embedding
        if self.embedding is not None:
            self.tensor_func = lambda x, emb: emb(Variable(t.LongTensor(x))).data
        else:
            self.tensor_func = lambda x: t.LongTensor(x)

    def __getitem__(self, index):
        xx = self.tensor_func(self.field.x[index], self.embedding.x_emb)
        yy = self.tensor_func(self.field.y[index], self.embedding.y_emb)
        return xx, yy

    def __len__(self):
        return self.field.length


class DataProcess(object):
    def __init__(self, field, embedding=None):
        self.data = DataSet(field, embedding=embedding)

    def generate_data(self, batch_size=1, shuffle=True, num_workers=0):
        self.loader = dataloader.DataLoader(self.data,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)
        return self.loader


class MyModel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=10,
                      kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        # self.conv = nn.DataParallel(self.conv)
        self.fc = nn.Linear(90, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        conv_out = self.conv(x)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc(reshaped)
        return logits


class Train(object):
    def __init__(self, dataset, epochs=1, batch_size=10, shuffle=True, data_num_workers=0, model_path=None):
        self.data_process = DataProcess(dataset)
        self.data_process.data_process()
        self.my_loader = data_process.generate_data(batch_size, shuffle, data_num_workers)
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_batch = self.my_loader.__len__()

    def model(self):
        if self.model_path is None:
            self.model = MyModel()
            if cuda_available():
                self.model = self.model.cuda()
        else:
            self.model = t.load(model_path)
            return self.model

    def optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        return self.optimizer

    def train(self, model):
        epoch_loss_meter = []
        for epoch in range(self.epochs):
            loss_meter = []
            t0 = datetime.datetime.now()
            for i, (x, y) in enumerate(self.my_loader):
                if cuda_available():
                    x = Variable(x).cuda()
                    y = Variable(y).cuda()
                else:
                    x = Variable(x)
                    y = Variable(y)
                optimizer.zero_grad()
                score = model(x)
                loss = loss_function(score, y)
                loss_meter.append(loss.data[0])
                loss.backward()
                optimizer.step()
                if i % 1000 == 1000 - 1:
                    time_delta = (datetime.datetime.now() - t0).seconds
                    print(epoch, i + 1, self.max_batch, time_delta, loss.data[0])
            epoch_loss_meter.append(loss_meter)
        return


if __name__ == '__main__':
    data_path = "./data/names"
    data_field = InputData(data_path).read()

    valid_loader = DataProcess(data_field["valid"], data_field["dict"]).generate_data(1,True)
    for i,(x,y) in enumerate(valid_loader):
        if i==2:
            print(y)
