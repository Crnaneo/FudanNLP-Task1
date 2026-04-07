import torch;
class Gram:
    def __init__(self,x,y,n):
        self.x = x;
        self.y = y;
        self.n = n;
        self.words = {};

    def get_words(self):
        for sentence in self.x:
            sentence = sentence.split(" ");
            for (i,word) in enumerate(sentence[:-(self.n-1)]):
                word = '_'.join([j.strip() for j in sentence[i:i+self.n]])
                if(word not in self.words):
                    self.words[word] = len(self.words);
        self.weight = torch.zeros(len(self.x),len(self.words));

    def get_matrix(self):
        for (i,sentence) in enumerate(self.x):
            sentence = sentence.split(" ");
            for (j,word) in enumerate(sentence[:-(self.n-1)]):
                word = '_'.join([j.strip() for j in sentence[i:i + self.n]])
                self.weight[i][self.words[word]]+=1;