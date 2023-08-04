import nltk 
import numpy as np # a python library used to perform arthimetic functions using python


from nltk.stem.porter import WordNetLemmatizer # library for stemming
#There are different stemmers and probably some other stemming libraries available 
# so you can try out different ones for yourself.

stemmer = PorterStemmer()
# the porter stemmer is assigned to a variable for easy accessibility

#FUNCTIONS FOR "TOKENIZATION", "STEMMING", AND "BAG OF WORDS"
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    # tokenization is for sentences. it helps to group the sentences 

def stem(word):
    return stemmer.stem(word.lower())
    # stemming is for words. it helps to group words

def bag_of_words(tokenized_sentence, all_words):
# the BOW will consist of the tokenized sentences and the stemmed words as shown
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

# We'll create a bag and initialize it with zero using Numpy

    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
 
    return bag





# EXAMPLE
#sentence = ["hello","how","are","you"]
#words = ['hey','hi','how','good','bye','you']
#bag = bag_of_words(sentence, words)
#print(bag)


#words = ["organize", "Organizes", "organizing"]
#stemmed_words = [stem(w) for w in words]
#print(stemmed_words)