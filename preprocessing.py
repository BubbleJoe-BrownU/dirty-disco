import numpy as np
import string
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
import pandas as pd
import collections

def tokenize(sentence):
    lemmatizer = WordNetLemmatizer()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation)).lower().split()
    for i in range(len(sentence)):
        sentence[i] = lemmatizer.lemmatize(sentence[i])
    return sentence

def preprocess(data):
    texts, labels = [], []
    data = list(data)
    for i in tqdm(range(len(data))):
        line = data[i]
        x = line["article"].numpy().decode('utf-8')
        y = line["highlights"].numpy().decode('utf-8')
        x = tokenize(x)
        y = tokenize(y)
        texts.append(x)
        labels.append(y)
        
    return texts, labels

def save_data(texts, labels, filepath='data/dataset'):
    df = pd.DataFrame(
        {
            "input" : texts,
            "label" : labels,
        }
    )
    df.to_csv(filepath, index=False)
    
def recover_from_dataframe(filepath="data/subset"):
    df = pd.read_csv(filepath)
    inputs = df["input"].str.split("', '")
    labels = df["label"].str.split("', '")


    for i in range(len(inputs)):
        inputs[i][0] = inputs[i][0][2:]
        inputs[i][-1] = inputs[i][-1][:-2]
        labels[i][0] = labels[i][0][2:]
        labels[i][-1] = labels[i][-1][:-2]
        
    return inputs, labels
    

def sentence_windowized(sentences, window_size=20):
    for i, sentence in enumerate(sentences):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa

#         # Convert the caption to lowercase, and then remove all special characters from it
#         caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption.lower())
      
        # Split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
#         clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
      
        # Join those words into a string

        sentence_new = ['<start>'] + sentence[:window_size-2] + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        sentences[i] = sentence_new
    return sentences



def load_data(filepath="data/subset", window_size=20):

    inputs, labels = recover_from_dataframe(filepath)
    
    labels = sentence_windowized(labels, window_size)
#     inputs = sentence_windowized(inputs, 500)
    inputs = sentence_windowized(inputs, 100)
    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for sent in labels:
        word_count.update(sent)
    for sent in inputs:
        word_count.update(sent)

    def unk_sentences(sentences, minimum_frequency):
        for sentence in sentences:
            for index, word in enumerate(sentence):
                if word_count[word] <= minimum_frequency:
                    sentence[index] = '<unk>'

    unk_sentences(labels, 80)
    unk_sentences(inputs, 80)

    # pad captions so they all have equal length
    def pad_captions(sentences, window_size):
        for sentence in sentences:
            sentence += (window_size - len(sentence)) * ['<pad>'] 
    
    pad_captions(labels,  window_size)
#     pad_captions(inputs, 500)
    pad_captions(inputs, 100) ## change it back to 500 when the architecture works

    # assign unique ids to every word left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for sent in inputs+labels:
        for index, word in enumerate(sent):
            if word in word2idx:
                sent[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                sent[index] = vocab_size
                vocab_size += 1
#     for sent in test_captions:
#         for index, word in enumerate(caption):
#             caption[index] = word2idx[word] 
    encoding = lambda x: [[word2idx[i] for i in line] for line in x]
    
    return dict(
        inputs = np.array(encoding(inputs)),
        labels = np.array(encoding(labels)),
        word2idx = word2idx,
        idx2word = {i:w for w, i in word2idx.items()}
    )
#     return dict(
#         train_captions          = np.array(train_captions),
#         test_captions           = np.array(test_captions),
#         train_image_features    = np.array(train_image_features),
#         test_image_features     = np.array(test_image_features),
#         train_images            = np.array(train_images),
#         test_images             = np.array(test_images),
#         word2idx                = word2idx,
#         idx2word                = {v:k for k,v in word2idx.items()},
#     )
