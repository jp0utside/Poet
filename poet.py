import nltk
from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm import MLE
from nltk.corpus import brown
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import re
import pronouncing

def tokenize_sent(sent):
    tokens = []
    sents = nltk.sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    for sent in sents:
        temp = tokenizer.tokenize(sent)
        temp.insert(0, "<s>")
        temp.append("</s>")
        tokens.append(temp)
    
    return tokens

#Originally designed for Dr. Seuss corpus
def get_bigrams(text):
    bigrams = []
    tokenizer = RegexpTokenizer(r'\w+')
    sents = nltk.sent_tokenize(text)
    for sent in sents:
        print(sent)
        words = tokenizer.tokenize(sent)
        print(words)
        padded = nltk.bigrams(pad_both_ends(words, n=2))
        bigrams.append(padded)
    for bigram in bigrams:
        for tup in bigram:
            print(tup)

#Function inspired by tutorial from https://www.nltk.org/api/nltk.lm.html
def sample_sentence(filename):
    file = open(filename, 'r')
    text = file.read()
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.lower()

    sents = nltk.sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    
    
    tokens = []
    for sent in sents:
        words = tokenizer.tokenize(sent)
        for word in words:
            if (len(pronouncing.phones_for_word(word)) == 0):
                words.remove(word)
        tokens.append(words)
    #print(tokens)

    
    train, vocab = padded_everygram_pipeline(3, tokens)

    lm = MLE(3)
    lm.fit(train, vocab)

    sent = []
    cur = "<s>"
    prev = "<s>"
    sent.append(cur)
    while cur != "</s>":
        seed = [prev, cur]
        #print(seed)
        nxt = lm.generate(1, text_seed = seed)
        #if(nxt != '<s>' and nxt != '</s>'):
            #while(len(pronouncing.phones_for_word(nxt)) == 0):
                #print(nxt)
                #nxt = lm.generate(1, text_seed = seed)
        prev = cur
        cur = nxt
        sent.append(cur)
    return sent

def sent_len(filename, syl):
    file = open(filename, 'r')
    text = file.read()
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.lower()

    sents = nltk.sent_tokenize(text)
    tokenizer = RegexpTokenizer(r'\w+')
    
    
    tokens = []
    for sent in sents:
        words = tokenizer.tokenize(sent)
        for word in words:
            if (len(pronouncing.phones_for_word(word)) == 0):
                words.remove(word)
        tokens.append(words)
    #print(tokens)

    
    train, vocab = padded_everygram_pipeline(3, tokens)

    lm = MLE(3)
    lm.fit(train, vocab)

    sent = []
    cur = "<s>"
    prev = "<s>"
    sent.append(cur)
    syl_count = 0
    while syllable_count(sent) < syl:
        seed = [prev, cur]
        #print(seed)
        nxt = lm.generate(1, text_seed = seed)
        #if(nxt != '<s>' and nxt != '</s>'):
            #while(len(pronouncing.phones_for_word(nxt)) == 0):
                #print(nxt)
                #nxt = lm.generate(1, text_seed = seed)
        if(nxt != "</s>" and syllable_count(sent) + syllable_count([nxt]) <= syl):
            prev = cur
            cur = nxt
            sent.append(cur)
    sent.append("</s>")
    return sent

def syllable_count(sent):
    count = 0
    for word in sent:
        if word != '<s>' and word != '</s>':
            plist = pronouncing.phones_for_word(word)
            count += pronouncing.syllable_count(plist[0])
    return count

def get_sent(file, syl, rhyme):
    sent = sent_len(file, syl)
    if rhyme != "":
        while rhyme not in pronouncing.rhymes(sent[-2]):
            sent = sent_len(file, syl)
    return sent

def get_sent_just_rhyme(file, syl, rhyme):
    sent = sample_sentence(file)
    if rhyme != "":
        while rhyme not in pronouncing.rhymes(sent[-2]):
            sent = sample_sentence(file)
    return sent


def generate_poem(file, syl, line):
    poem = []
    last = ["",""]
    for i in range(line):
        cur = get_sent_just_rhyme(file, syl, last[-2])
        poem.append(cur)
        last = cur
    return poem

def generate_haiku(file):
    haiku = []
    first = sample_sentence(file)
    second = sample_sentence(file)
    third = sample_sentence(file)
    while(syllable_count(first) != 5):
        first = sample_sentence(file)
    while(syllable_count(second) != 7):
        second = sample_sentence(file)
    while(syllable_count(third) != 5):
        third = sample_sentence(file)
    haiku.append(first)
    haiku.append(second)
    haiku.append(third)
    return haiku

    
def print_poem(poem):
    string = ""
    for line in poem:
        for word in line:
            if word != "<s>" and word != "</s>":
                string += word + " "
        print(string)
        string = ""
    
"""
Dr. Seuss corpus borrowed from Roberts Dionne GitHub https://github.com/robertsdionne
Corpus is a compilation of the following Dr. Seuss stories, cleaned appropriately:
    The Cat in the Hat
    Fox in Socks
    Green Eggs and Ham
    How the Grinch Stole Christmas
    Hop on Pop
    One Fish Two Fish
"""

def main():
    file = "Seuss.txt"
    sent = sample_sentence(file)
    poem = generate_haiku(file)
    print_poem(poem)


    #for sent in sentences:
        #print(sent)
if __name__ == "__main__": main()
