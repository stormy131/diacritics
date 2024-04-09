import numpy as np
import torch
import torch.nn as nn
from data_utils import Dataset, Encoder
from modelling import MLP, train, test

N_GRAMS = 3
MLP_HIDDEN = 16
DIA_CLASSES = 3

# Get train data
dataset = Dataset()
data_encoder = Encoder(N_GRAMS)

# Get validation/test data
with open('data/eval/diacritics-dtest.txt',mode='r') as val_file, open('data/eval/diacritics-etest.txt', mode='r') as test_file:
    val_text, test_text = val_file.read(), test_file.read()

# Data preprocessing
data, target = data_encoder.get_model_data(dataset.target)
v_data, v_target = data_encoder.get_model_data(val_text)
t_data, t_target = data_encoder.get_model_data(test_text)

train_data, train_target = torch.from_numpy(data), torch.from_numpy(target).type(torch.LongTensor)
val_data, val_target = torch.from_numpy(v_data), torch.from_numpy(v_target).type(torch.LongTensor)
test_data, test_target = torch.from_numpy(t_data), torch.from_numpy(t_target).type(torch.LongTensor)

# Modelling
model = MLP(len(data[0]), MLP_HIDDEN, DIA_CLASSES)
train(model, train_data, train_target)
   
print(f'VALIDATION ACCURACY - {test(model, val_data, val_target):.2f}')
print(f'TEST ACCURACY - {test(model, test_data, test_target)}')