import re
import os
import urllib.request as req
import numpy as np

LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_DIA = 'áčďéěíňóřšťúůýž'

# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

class Encoder:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.acute = 'áéíóúý'
        self.caron = 'čďěňřšťůž'
        self.num_char = 26
        
        
    def _ohe_chars(self, s: str) -> np.ndarray:
        res = np.zeros(shape=(len(s), self.num_char))
        for i in range(len(s)):
            char_value = ord(s[i].lower()) - ord('a')
            if char_value < 0 or char_value >= self.num_char:
                continue
            
            res[i, char_value] = 1
            
        return res
        
    
    # Window applied only to current word
    # def make_train(self, text: str) -> np.ndarray:
    #     res = np.zeros(shape=(len(text), self.window_size, self.num_char))
    #     words = re.split(r'\W+', text)
    #     for w in words:
    #         window_offset = (self.window_size - 1) // 2
    #         ohes = self.ohe_chars('.'*window_offset + w + '.'*window_offset)
    
    def get_model_data(self, text: str) -> tuple[np.array, np.array]:
        no_dia = text.translate(DIA_TO_NODIA)
        data, targets = self._process_data(no_dia), self._process_targets(text)
        
        return np.array([w.flatten() for w in data]), np.array(targets)
        
    
    # Window applied to all subsequent letters in text
    def _process_data(self, text: str) -> np.ndarray:
        res = np.zeros(shape=(len(text), self.window_size, self.num_char),
                       dtype=np.float32)
        window_offset = (self.window_size - 1) // 2
        ohes = self._ohe_chars('.'*window_offset + text + '.'*window_offset)

        for i in range(len(text)):
            res[i] = ohes[i:i + 2*window_offset + 1]
            
        return res
    
    
    def _process_targets(self, text: str) -> np.ndarray:
        res = np.zeros(shape=len(text))
        for i in range(len(text)):
            if text[i] in self.acute:
                res[i] = 1
            elif text[i] in self.caron:
                res[i] = 2
                
        return res


# CODE SNIPPET FROM NPFL129 COURSE
class Dataset:
    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        
        if not os.path.exists(f'./data/train/{name}'):
            print('Downloading NPFL129 data')
            licence_name = name.replace(".txt", ".LICENSE")
            req.urlretrieve(url + licence_name, filename=f'./data/train/{licence_name}')
            req.urlretrieve(url + name, filename="./data/train/{}.tmp".format(name))
            os.rename("./data/train/{}.tmp".format(name), f'./data/train/{name}')

        # Load the dataset and split it into `data` and `target`.
        with open(f'./data/train/{name}', "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()