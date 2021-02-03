import json
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import nltk
from nltk.stem.lancaster import LancasterStemmer
# nltk.download('punkt')
stemmer = LancasterStemmer()

with open('intents.json') as f:
    data = json.load(f)

words = []  # store all words in data  # prepare for bag of words  # Features
intents = []  # store all intents  # Labels
document = []  # store words_in_pattern and intent pairs
words_to_ignore = ['?', '.']

for intent in data['intents']:
    for pattern in intent['patterns']:
        words_each_pattern = nltk.word_tokenize(pattern)  # split the sentence
        words.extend(words_each_pattern)  # w is a list
        document.append((words_each_pattern, intent['tag']))

    if intent['tag'] not in intents:
        intents.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in words_to_ignore]  # stem a word means to find the root of the word

words = sorted(list(set(words)))
intents = sorted(list(set(intents)))

X = []  # features
y = []  # labels

output = [0 for _ in range(len(intents))]

for doc in document:
    bag = []  # len(bag) == len(words_all)  # words --> digit  # 1 for the word exists, 0 for not

    words_each_pattern = doc[0]
    words_each_pattern = [stemmer.stem(w.lower()) for w in words_each_pattern]

    for w in words:
        if w in words_each_pattern:
            bag.append(1)
        else:
            bag.append(0)
    
    out = output[:]
    out[intents.index(doc[1])] = 1
    
    X.append(bag)
    y.append(out)

X = np.array(X)
y = np.array(y)

# Note:
## Softmax function is used to normalize the outputs, converting them from weighted sum values into probabilities that sum to one.

in_size = len(X[0])
out_size = len(y[0])

model = keras.Sequential([
    layers.Dense(20, activation='relu', input_shape=(in_size,)),
    layers.Dense(10, activation='relu'),
    layers.Dropout(0.1),  # Dropout is used for preventing overfit
    layers.Dense(out_size, activation='softmax')
])
    
opt = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X, y, epochs=200, batch_size=5)

model.save('model.h5')

# inp = pd.DataFrame([X[0]])
# prediction = model.predict(inp)
# print(prediction)

