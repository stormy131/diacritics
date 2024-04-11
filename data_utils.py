import re
import os
import urllib.request as req
import numpy as np
import torch
import torch.utils.data as data

LETTERS_NODIA = 'acdeeinorstuuyz' + 'acdeeinorstuuyz'.upper()
LETTERS_DIA = 'áčďéěíňóřšťúůýž' + 'áčďéěíňóřšťúůýž'.upper()

# A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
DIA_TO_NODIA = str.maketrans(LETTERS_DIA, LETTERS_NODIA)

class TextData(data.Dataset):
    def __init__(self, data: torch.Tensor, target: torch.Tensor) -> None:
        self.dataset = data
        self.target = target
    
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    
    def __getitem__(self, idx: list[int]) -> torch.Tensor:
        return self.dataset[idx].type(torch.float32), self.target[idx].type(torch.LongTensor)


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
    
    # Window applied to subsequent letters in word ONLY    
    def process_text(self, text: str) -> tuple[np.array, np.array]:
        res = []
        target = []
        window_offset = (self.window_size - 1) // 2
        
        for word in re.split(r'\W+', text):
            word_no_dia = word.translate(DIA_TO_NODIA)
            word_ohe = self._ohe_chars('.'*window_offset + word_no_dia + '.'*window_offset)
            
            for i in range(len(word)):
                if word_no_dia[i] not in LETTERS_NODIA:
                    continue
                
                if word[i] in LETTERS_NODIA:
                    target.append(0)
                elif word[i] in self.acute:
                    target.append(1)
                else:
                    target.append(2)
                
                res.append(word_ohe[i:i + 2*window_offset + 1].flatten())
                
        return np.array(res), np.array(target)


    # Window applied to all subsequent letters in text
    # def process_text(self, text: str) -> tuple[np.array, np.array]:
    #     res = []
    #     target = []
    #     no_dia = text.translate(DIA_TO_NODIA)
        
    #     window_offset = (self.window_size - 1) // 2
    #     ohes = self._ohe_chars('.'*window_offset + no_dia + '.'*window_offset)

    #     for i in range(len(text)):
    #         if no_dia[i] not in LETTERS_NODIA:
    #             continue
            
    #         if text[i] in LETTERS_NODIA:
    #             target.append(0)
    #         elif text[i] in self.acute:
    #             target.append(1)
    #         else:
    #             target.append(2)
            
    #         res.append(ohes[i:i + 2*window_offset + 1].flatten())
            
    #     return np.array(res), np.array(target)


# CODE SNIPPET FROM NPFL129 COURSE
class NPFL129_Dataset:
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
            
if __name__ == '__main__':
    d = NPFL129_Dataset()
    e = Encoder(5)
    data, target = e.process_text(d.target)