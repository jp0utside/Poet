import nltk
from nltk.corpus import brown
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
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import datetime


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
    file.close()
    return tokens

def clean_sents(sents):
    new_sents = []
    for sent in sents:
        news = []
        for word in sent:
            new = word.lower()
            new.replace("'", "")
            new.replace('"', "")
            news.append(new)
        new_sents.append(news)
    return new_sents
    

#Function inspired by tutorial from https://www.nltk.org/api/nltk.lm.html
def sample_sentence(lm):
    sent = []
    cur = "<s>"
    prev = "<s>"
    sent.append(cur)
    while cur != "</s>":
        seed = [prev, cur]
        nxt = lm.generate(1, text_seed = seed)
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

def sample_sentence_lim(lm, syl):
    sent = []
    cur = "<s>"
    prev = "<s>"
    sent.append(cur)
    rands = [0,1,2,3]
    loop = True
    seed = [prev, cur]
    sent = lm.generate(syl - random.choice(rands), text_seed = seed)
    return sent


def syllable_count(sent):
    count = 0
    for word in sent:
        if word != '<s>' and word != '</s>' and len(pronouncing.phones_for_word(word)) != 0:
            plist = pronouncing.phones_for_word(word)
            count += pronouncing.syllable_count(plist[0])
    return count

"""
Function to generate a haiku.
"""
def haiku(lm):
    haiku = []
    first = sample_sentence_syl(lm, 5)
    second = sample_sentence_syl(lm, 7)
    third = sample_sentence_syl(lm, 5)
    while(syllable_count(first) != 5):
        first = sample_sentence_syl(lm, 5)
    while(syllable_count(second) != 7):
        second = sample_sentence_syl(lm, 7)
    while(syllable_count(third) != 5):
        third = sample_sentence_syl(lm, 5)
    haiku.append(first)
    haiku.append(second)
    haiku.append(third)
    return haiku

"""
Function to generate a poem with an inputted amount of lines and lines that rhyme, not allowing lines to end with
the same word.
"""
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
                        if(len(i) == lines):
                            poem = i
                            cont = False
        if not put:
            q = deque()
            q.append(sent)
            poems.append(q)
    return poem

"""
Function to generate a poem with an inputted amount of syllables per line,
an inputted amount of lines, and lines that rhyme, not allowing lines to end with
the same word.
"""
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

"""
Function to generate a poem with an inputted amount of syllables per line,
an inputted amount of lines, and lines that rhyme, allowing for lines to end with
the same word.
"""
def poem_search_syl_reprhyme(lm, syl, lines):
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
                    put = True
                    i.append(sent)
                    if(len(i) == lines):
                        poem = i
                        cont = False
        if not put:
            q = deque()
            q.append(sent)
            poems.append(q)
    return poem

"""
Function to generate a poem with an inputted amount of syllables per line,
an inputted amount of lines, and an ABAB rhyming scheme.
"""
def poem_search_syl_abab(lm, syl, lines):
    poem = []
    first = poem_search_syl(lm, syl, lines)
    second = poem_search_syl(lm, syl, lines)
    for i in range(len(first)):
        poem.append(first[i])
        poem.append(second[i])
    return poem

"""
Function to generate a poem with a common theme, an inputted amount of syllables per line,
and an inputted amount of lines.
"""
def poem_search_theme(lm, syl, lines, vec):
    poem = []
    first = sample_sentence_lim(lm, syl)
    poem.append(first)
    for i in range(lines):
        best = []
        score = 0
        for j in range(100):
            candidate = sample_sentence_lim(lm, syl)
            c_score = sent_eval(poem[-1], candidate, vec)
            if c_score > score:
                best = candidate
                score = c_score
        poem.append(best)
    return poem

"""
Function to generate a poem with a common theme, rhyming lines,
and a certain amount of syllables and lines.
"""
def poem_search_theme_rhyme(lm, syl, lines, vec):
    poem = []
    scores = []
    samples = poem_search_syl(lm, syl, lines*3)
    poem.append(samples[0])
    for k in range(1, lines):
        poem.append(samples[k])
        scores.append(sent_eval(poem[0], samples[k], vec))
    
    for i in range(lines, len(samples)):
        candidate = samples[i]
        c_score = sent_eval(poem[0], candidate, vec)
        for j in range(len(scores)):
            if c_score > scores[j]:
                scores[j] = c_score
                poem[j+1] = candidate
                break
    return poem

"""
Function to evaluate word to vector model similarity between each word in the two input strings.
"""
def sent_eval(first, second, vector_model):
    tot = 0
    count = 0
    for f in first:
        if f in vector_model.wv.index_to_key:
            for s in second:
                if s in vector_model.wv.index_to_key:
                    count += 1
                    tot += vector_model.wv.similarity(f, s)
    if tot == 0 or count == 0:
        avg = float('-inf')
    else:
        avg = tot/count
    return avg

"""
Function to take in array of lines, ignore start and end chars, and print lines like a sentence.
"""
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
    start_time = datetime.datetime.now()
    print(start_time)
    file = "Whitman.txt"
    sent_tokens = tokenize_text(file)
    #sent_tokens = clean_sents(brown.sents(categories = "news"))

    train, vocab = padded_everygram_pipeline(3, sent_tokens)
    lm = MLE(3)
    lm.fit(train, vocab)

    stop_words = set(stopwords.words('english'))
    word_tokens = []
    filtered_tokens = []
    for sent in sent_tokens:
        new_sent = []
        for word in sent:
            word_tokens.append(word)
            if word not in stop_words:
                new_sent.append(word)
            filtered_tokens.append(new_sent)
    

    #vector_model = gensim.models.Word2Vec(filtered_tokens, min_count = 5)
    #vector_model.save("w2v_whitman.model")
    vector_model = Word2Vec.load("w2v_whitman.model")

    print_poem(haiku(lm))
    print()
    print_poem(poem_search(lm, 5))
    print()
    print_poem(poem_search_syl(lm, 10, 5))
    print()
    print_poem(poem_search_syl_abab(lm, 10, 2))
    print()
    print_poem(poem_search_theme(lm, 10, 5, vector_model))
    print()
    print_poem(poem_search_theme_rhyme(lm, 10, 3, vector_model))
    print()
    print(datetime.datetime.now() - start_time)
if __name__ == "__main__": main()
