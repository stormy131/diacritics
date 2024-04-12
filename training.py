import os 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_utils import NPFL129_Dataset, Encoder, DIA_TO_NODIA, TextData, LETTERS_NODIA
from mlp import MLP, train, test

MODEL_FILE = './model.pt'
N_GRAMS = 9
MLP_HIDDEN = 128
DIA_CLASSES = 3
dataset = NPFL129_Dataset()
data_encoder = Encoder(N_GRAMS)
RESTORE_MAP = {
    1: {c.translate(DIA_TO_NODIA): c for c in data_encoder.acute + data_encoder.acute.upper()},
    2: {c.translate(DIA_TO_NODIA): c for c in data_encoder.caron + data_encoder.caron.upper()}
}

def restore_dia(origin_text: str, data: torch.Tensor, model: nn.Module) -> str:
    prediction = torch.argmax(model(data.type(torch.float32)), dim=1)
    pred_iter = iter(prediction.tolist())

    restored = ""
    for char in origin_text.translate(DIA_TO_NODIA):
        if char in LETTERS_NODIA:
            predicted = next(pred_iter)
            if predicted == 0 or char not in RESTORE_MAP[predicted]:
                restored += char
            else:
                restored += RESTORE_MAP[predicted][char]
        else:
            restored += char
            
    return restored

# Get train data
if __name__ == '__main__':
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
    model = MLP(N_GRAMS * data_encoder.num_char, MLP_HIDDEN, DIA_CLASSES)
    if os.path.exists(MODEL_FILE):
        model.load_state_dict(torch.load(MODEL_FILE))
    else:
        print('TRAINING MODEL')
        
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
    model.load_state_dict(best_state)
    restored = restore_dia(test_text, test_data, model)
    correct = 0
    for predicted, original in zip(restored, test_text):
        if str.isspace(original):
            continue
        
        correct += 1 if predicted == original else 0

    print(f'\nTEST ACCURACY - {correct / len([x for x in test_text if not str.isspace(x)]):.2f}')