# pytorch-learning
Learning PyTorch by NLP Tasks

Learn pytorch with the following repos:
* [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* [practical-pytorch](https://github.com/spro/practical-pytorch)

The learning steps are as follows:
1. follow the practical-pytorch step by step, organize code in the way of OpenNMT-py
2. write my own NMT code, or more simply, write a self-designed NN to clean bilingual text data

A machine learning task should include these steps:
* data processing
* training
* prediction

While doing data processing, we should decode the data into recognizable datasets,
clean the data, split data into train, valid, test datasets, and make relevant features.
To improve the training quality, sometimes we have to do data screening.
This is the step of understanding data.

In NLP tasks, we have to process text data, sound data even image data.
As for text data, the most important thing is text cleaning and screening.
In the learning, we do not focus on this topic, instead, we will pay more attention on word/character embedding.

While training a NN, we should construct the model well and tune the parameters well.
The hardest part is code designing, once wrote done code, the only thing we have to do is tuning the parameters.

There are many good github repos such as OpenNMT-py and practical-pytorch,
from which we can learn how to orginize our code, what's more,
we can improve our python skills and deep-learning skills by learning those repos.

Prediction is much more complicate, it requires deep data understanding and we will skip this topic.

All codes must be written by myself, never use the classes and functions in the referring repos.

## Classifying Names
download data from practical-pytorch/data
```
python trainvalidsplit.py
python preprocess.py
python train.py
python predict.py
```

* Only use official python packages in trainvalidsplit.py, so it is quite simple. For the sake of improving performance, we can use pandas, numpy and concurrent





## Generating Shakespeare

## Generating Names

## Translation

## Exploring Word Vectors

