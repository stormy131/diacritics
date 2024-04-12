import sys
import torch
from mlp import MLP
from training import N_GRAMS, MLP_HIDDEN, DIA_CLASSES, data_encoder, MODEL_FILE, restore_dia

model = MLP(N_GRAMS * data_encoder.num_char, MLP_HIDDEN, DIA_CLASSES)
model.load_state_dict(torch.load(MODEL_FILE))

input_text = ''.join(sys.stdin.readlines())
data_encoding, _ = data_encoder.process_text(input_text)

print('\n---> Restored:')
print(restore_dia(input_text, torch.from_numpy(data_encoding), model))