import os 
import torch
from torch.utils.data import DataLoader
from data_utils import NPFL129_Dataset, Encoder, DIA_TO_NODIA, TextData, LETTERS_NODIA
from modelling import MLP, train, test

MODEL_FILE = './model.pt'
N_GRAMS = 13
MLP_HIDDEN = 128
DIA_CLASSES = 3

# Get train data
dataset = NPFL129_Dataset()
data_encoder = Encoder(N_GRAMS)
RESTORE_MAP = {
    1: {c.translate(DIA_TO_NODIA): c for c in data_encoder.acute + data_encoder.acute.upper()},
    2: {c.translate(DIA_TO_NODIA): c for c in data_encoder.caron + data_encoder.caron.upper()}
}

# Get validation/test data
with open('data/eval/diacritics-dtest.txt',mode='r') as val_file, open('data/eval/diacritics-etest.txt', mode='r') as test_file:
    val_text, test_text = val_file.read(), test_file.read()

# Data preprocessing
data, target = data_encoder.process_text(dataset.target)
v_data, v_target = data_encoder.process_text(val_text)
t_data, t_target = data_encoder.process_text(test_text)

train_data, train_target = torch.from_numpy(data), torch.from_numpy(target)
val_data, val_target = torch.from_numpy(v_data), torch.from_numpy(v_target)
test_data, test_target = torch.from_numpy(t_data), torch.from_numpy(t_target)

# Modelling
model = MLP(len(data[0]), MLP_HIDDEN, DIA_CLASSES)

if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE))
else:
    best_eval, best_state = 0, None
    for e in range(5):
        print(f'\nEpoch {e}')
        train(model, DataLoader(TextData(train_data, train_target), batch_size=128, shuffle=True))
        test_eval = test(model, val_data, val_target)
        if test_eval > best_eval:
            best_eval, best_state = test_eval, model.state_dict()
        
        print(f'VALIDATION ACCURACY - {test_eval:.2f}')

    # model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_FILE)

# Restoring
prediction = torch.argmax(model(test_data.type(torch.float32)), dim=1)
pred_iter = iter(prediction.tolist())

correct = 0
restored = ""
for char, actual in zip(test_text.translate(DIA_TO_NODIA), test_text):
    if char in LETTERS_NODIA:
        predicted = next(pred_iter)
        if predicted == 0 or char not in RESTORE_MAP[predicted]:
            restored += char
        else:
            restored += RESTORE_MAP[predicted][char]
            
        correct += 1 if restored[-1] == actual else 0
    else:
        restored += char
        correct += 0 if str.isspace(char) else 1

print(f'TEST ACCURACY - {correct / len([x for x in test_text if not str.isspace(x)])}')
print(restored[:300])