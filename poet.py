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
import random
from collections import deque

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

def tokenize_text(filename):
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
            if not pronouncing.phones_for_word(word):
                words.remove(word)
            elif len(word) == 1 and word != "a" and word != "i":
                words.remove(word)
        tokens.append(words)
    return tokens


#Function inspired by tutorial from https://www.nltk.org/api/nltk.lm.html
def sample_sentence(lm):

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

def sample_sentence_syl(lm, syl):

    sent = []
    cur = "<s>"
    prev = "<s>"
    sent.append(cur)
    rands = [0,1,2,3]
    loop = True
    while loop:
        seed = [prev, cur]
        sent = lm.generate(syl - random.choice(rands), text_seed = seed)
        if(syllable_count(sent) == syl):
            loop = False
    return sent

def syllable_count(sent):
    count = 0
    for word in sent:
        if word != '<s>' and word != '</s>' and len(pronouncing.phones_for_word(word)) != 0:
            plist = pronouncing.phones_for_word(word)
            count += pronouncing.syllable_count(plist[0])
    return count

def get_sent(lm, syl, rhyme):
    sent = sample_sentence(lm)
    if rhyme != "":
        while rhyme not in pronouncing.rhymes(sent[-2]):
            sent = sample_sentence(lm)
            print(rhyme)
            print(sent)
    return sent


def generate_poem(lm, syl, line):
    poem = []
    last = ["",""]
    for i in range(line):
        cur = get_sent(lm, syl, last[-2])
        poem.append(cur)
        last = cur
    return poem

def generate_haiku(lm):
    haiku = []
    first = sample_sentence_syl(lm, 5)
    second = sample_sentence_syl(lm, 7)
    third = sample_sentence_syl(lm, 5)
    while(syllable_count(first) != 5):
        print("getting new first")
        first = sample_sentence_syl(lm, 5)
    while(syllable_count(second) != 7):
        print("getting new second")
        second = sample_sentence_syl(lm, 7)
    while(syllable_count(third) != 5):
        print("getting new third")
        third = sample_sentence_syl(lm, 5)
    haiku.append(first)
    haiku.append(second)
    haiku.append(third)
    return haiku

def poem_search(lm, lines):
    poems = [deque()]
    poem = []
    cont = True
    while cont:
        sent = sample_sentence(lm)
        put = False
        for i in poems:
            if len(i) > 0:
                if i[-1][-2] in pronouncing.rhymes(sent[-2]):
                    pres = False
                    put = True
                    for j in i:
                        if j[-2] == sent[-2]:
                            pres = True
                    if not pres:
                        i.append(sent)
                        print(i)
                        if(len(i) == lines):
                            poem = i
                            cont = False
        if not put:
            q = deque()
            q.append(sent)
            poems.append(q)
    return poem

def poem_search_syl(lm, syl, lines):
    poems = [deque()]
    poem = []
    cont = True
    while cont:
        sent = sample_sentence_syl(lm, syl)
        while sent[-1] == "</s>" or sent[-1] =="<s>":
            sent.pop()
        put = False
        for i in poems:
            if len(i) > 0:
                if i[-1][-1] in pronouncing.rhymes(sent[-1]):
                    pres = False
                    put = True
                    for j in i:
                        if j[-1] == sent[-1]:
                            pres = True
                    if not pres:
                        i.append(sent)
                        if(len(i) == lines):
                            poem = i
                            cont = False
        if not put:
            q = deque()
            q.append(sent)
            poems.append(q)
    return poem

def poem_search_syl_abab(lm, syl, lines):
    poem = []
    first = poem_search_syl(lm, syl, lines)
    second = poem_search_syl(lm, syl, lines)
    for i in range(len(first)):
        poem.append(first[i])
        poem.append(second[i])
    return poem
    
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
    tokens = tokenize_text(file)
    train, vocab = padded_everygram_pipeline(3, tokens)
    lm = MLE(3)
    lm.fit(train, vocab)

    poem = poem_search_syl_abab(lm, 5, 5)
    print_poem(poem)


    #for sent in sentences:
        #print(sent)
if __name__ == "__main__": main()
