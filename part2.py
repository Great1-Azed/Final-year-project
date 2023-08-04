import json
# this is a json module that helps to perform some certain json functions
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from part1 import tokenize, stem, bag_of_words 
#this helps us to re use already written code

from part3 import NeuralNet


with open('intents.json', 'r') as x:
    intents = json.load(x)
# code above opens, and loads the Json file that we have created


all_words = []
tags = []
xy = []
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
#in the above code, list comprehension was applied.


all_words = sorted(set(all_words))
tags = sorted(set(tags))
#the set technique helps to remove duplicate elements.
print(tags)

#implementing bag_of_words
# we'll create empty lists for the training data
x_train = []
y_train = []
for (pattern_sentence, tags) in xy: #looping over the array of xy
    bag = bag_of_words(pattern_sentence, all_words)  
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append (label) #crossEntropyLoss

    # we want to import the vectos abaove as a numpy array, so we import numpy
x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self) :
        self.num_of_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    #dataset[index]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.num_of_samples


#Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])

print(input_size, len(all_words))
print(output_size, tags)




dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# we create our training files after we're done with part 3



model = NeuralNet(input_size, hidden_size, output_size)
