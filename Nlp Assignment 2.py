#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:17:02 2019

@author: areebwaseem
"""



##
import os.path
import sys
import random
from operator import itemgetter
from collections import defaultdict
import math
#----------------------------------------
#  Data input 
#----------------------------------------

# Read a text file into a corpus (list of sentences (which in turn are lists of words))
# (taken from nested section of HW0)
def readFileToCorpus(f):
    """ Reads in the text file f which contains one sentence per line.
    """
    if os.path.isfile(f):
        file = open(f, "r") # open the input file in read-only mode
        i = 0 # this is just a counter to keep track of the sentence numbers
        corpus = [] # this will become a list of sentences
        print("Reading file ", f)
        for line in file:
            i += 1
            sentence = line.split() # split the line into a list of words
            #append this lis as an element to the list of sentences
            corpus.append(sentence)
            if i % 1000 == 0:
    	#print a status message: str(i) turns int i into a string
    	#so we can concatenate it
                sys.stderr.write("Reading sentence " + str(i) + "\n")
        #endif
    #endfor
        return corpus
    else:
    #ideally we would throw an exception here, but this will suffice
        print("Error: corpus file ", f, " does not exist")
        sys.exit() # exit the script
    #endif
#enddef


# Preprocess the corpus
def preprocess(corpus):
    #find all the rare words
    freqDict = defaultdict(int)
    for sen in corpus:
	    for word in sen:
	       freqDict[word] += 1
	#endfor
    #endfor

    #replace rare words with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if freqDict[word] < 2:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor

    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor
    
    return corpus
#enddef

def preprocessTest(vocab, corpus):
    #replace test words that were unseen in the training with unk
    for sen in corpus:
        for i in range(0, len(sen)):
            word = sen[i]
            if word not in vocab:
                sen[i] = UNK
	    #endif
	#endfor
    #endfor
    
    #bookend the sentences with start and end tokens
    for sen in corpus:
        sen.insert(0, start)
        sen.append(end)
    #endfor

    return corpus
#enddef

# Constants 
UNK = "UNK"     # Unknown word token
start = "<s>"   # Start-of-sentence token
end = "</s>"    # End-of-sentence-token


#--------------------------------------------------------------
# Language models and data structures
#--------------------------------------------------------------

# Parent class for the three language models you need to implement
class LanguageModel:
    # Initialize and train the model (ie, estimate the model's underlying probability
    # distribution from the training corpus)
    def __init__(self, corpus):
        print("""Your task is to implement four kinds of n-gram language models:
      a) an (unsmoothed) unigram model (UnigramModel)
      b) a unigram model smoothed using Laplace smoothing (SmoothedUnigramModel)
      c) an unsmoothed bigram model (BigramModel)
      d) a bigram model smoothed using kneser-ney smoothing (SmoothedBigramModelKN)
      """)
    #enddef

    # Generate a sentence by drawing words according to the 
    # model's probability distribution
    # Note: think about how to set the length of the sentence 
    #in a principled way
    def generateSentence(self):
        print("Implement the generateSentence method in each subclass")
        return "mary had a little lamb ."
    #emddef

    # Given a sentence (sen), return the probability of 
    # that sentence under the model
    def getSentenceProbability(self, sen):
        print("Implement the getSentenceProbability method in each subclass")
        return 0.0
    #enddef

    # Given a corpus, calculate and return its perplexity 
    #(normalized inverse log probability)
    def getCorpusPerplexity(self, corpus):
        print("Implement the getCorpusPerplexity method")
        return 0.0
    #enddef

    # Given a file (filename) and the number of sentences, generate a list
    # of sentences and write each to file along with its model probability.
    # Note: you shouldn't need to change this method
    def generateSentencesToFile(self, numberOfSentences, filename):
        filePointer = open(filename, 'w+')
        for i in range(0,numberOfSentences):
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
            
	#endfor
    #enddef
#endclass

# Unigram language model
class UnigramModel(LanguageModel):
    
    def __init__(self, corpus, test_corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        self.test_corpus = test_corpus
        
        
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef
    # Returns the probability of word in the distribution
    def prob(self, word):
        
        return (self.counts[word])/(self.total)
    
    def getSentenceProbability(self, sen):
        
         prob_found = 1
        
         for idx, word in enumerate(sen):
              if word == start:
                    continue
              prob_found = prob_found * self.prob(word)
             
         return prob_found
        
    #enddef
    
    def generateSentence(self):
        
        init_word = start
        final_sentence = []
        final_sentence.append(init_word)
        
        while(init_word != end):
            #rand = random.randint(0,len(self.counts))
            word = random.choice(list(self.counts))
            init_word = word
            final_sentence.append(init_word)
        
        
      #  print(final_sentence)
        return final_sentence
        
        
        
    #emddef
    def generateSentencesToFile(self, numberOfSentences, filename):
        
        filePointer = open(filename, 'w+')
        
        for i in range(0,numberOfSentences):
            
            sen = self.generateSentence()
            prob = self.getSentenceProbability(sen)

            stringGenerated = str(prob) + " " + " ".join(sen) 
            print(stringGenerated, end="\n", file=filePointer)
        
    def getCorpusPerplexity(self, test_corpus):
        words_num = 0
        perplex = 0
        for each_sentence in test_corpus:
            words_num = words_num + len(each_sentence)
            perplex = perplex + math.log(self.getSentenceProbability(each_sentence),2)
        
        perplexity = 2**((-1/words_num)*perplex) 
        return perplexity
    #endddef
#endclass

#Smoothed unigram language model (use laplace for smoothing)
class SmoothedUnigramModel(LanguageModel):
    
    vocab_size = 0
    def __init__(self, corpus,vocab,test_corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
        self.vocab_size = len(vocab)
        self.test_corpus = test_corpus
        
        
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef
    # Returns the probability of word in the distribution
    def prob(self, word):
        return (self.counts[word]+1)/(self.total + self.vocab_size)
    
    
    
    def getSentenceProbability(self, sen):
        
         prob_found = 1
        
         for idx, word in enumerate(sen):
              if word == start:
                    continue
              prob_found = prob_found * self.prob(word)
                
         return prob_found
     
        
    def generateSentence(self):
        
        init_word = start
        final_sentence = []
        final_sentence.append(init_word)
        
        while(init_word != end):
            #rand = random.randint(0,len(self.counts))
            word = random.choice(list(self.counts))
            init_word = word
            final_sentence.append(init_word)
        
       # print(final_sentence)
        return final_sentence
        
        
     
    def getCorpusPerplexity(self, test_corpus):
        words_num = 0
        perplex = 0
        for each_sentence in test_corpus:
            words_num = words_num + len(each_sentence)
            perplex = perplex + math.log(self.getSentenceProbability(each_sentence),2)
        
        perplexity = 2**((-1/words_num)*perplex) 
        return perplexity
    #endddef
#endclass

# Unsmoothed bigram language model
class BigramModel(LanguageModel):
    
    vocab_size = 0
    
    def __init__(self, corpus,vocab,test_corpus):
        
        self.counts = defaultdict(float)
        self.pairCounts = defaultdict(float)
        self.total = 0.0
        self.back_triple_sets = dict()
        self.train(corpus)
        self.vocab_size = len(vocab)
        self.test_corpus = test_corpus
        
        
    def train(self, corpus):
        for sen in corpus:
            for idx, word in enumerate(sen):
                self.counts[word] += 1.0
                self.total += 1.0
                if(idx > 0):
                    
                    self.pairCounts[(sen[idx-1],word)] += 1
                    
                    if(sen[idx - 1] in self.back_triple_sets):
                        
                        b = self.back_triple_sets[sen[idx - 1]]
                        b.add(word)
                        self.back_triple_sets[sen[idx - 1]] = b
                    else:
                        b = set()
                        b.add(word)
                        self.back_triple_sets[sen[idx - 1]] = b
            #endfor
        
        #endfor
    #enddef
    # Returns the probability of word in the distribution
    def prob(self, word_pair):
        word0, word1 = word_pair
        tuple_found = (word0,word1)
        return (self.pairCounts[tuple_found]/ self.counts[word0])
    
    
    def getSentenceProbability(self, sen):
        
         prob_found = 0
         is_first = True
         for idx, word in enumerate(sen):
             if(idx != 0):
                 if(is_first):
                     prob_found = self.prob((sen[idx-1],word))
                     is_first = False
                 else:
                     prob_found = prob_found * self.prob((sen[idx-1],word))
             
         return prob_found
     
    def generateSentence(self):
        
        init_word = start
        
        final_sentence = []
        
        final_sentence.append(init_word)
        
        while(init_word != end):
            #rand = random.randint(0,len(self.counts))
            next_options = self.back_triple_sets[init_word]
            selected_option = random.choice(list(next_options))
            #print("here")
           # print(selected_option)
            #print("next_option", selected_option)
            init_word = selected_option
            final_sentence.append(init_word)
        
        #print(final_sentence)
        return final_sentence
    
      
    
        
    def getCorpusPerplexity(self, test_corpus):
        words_num = 0
        perplex = 0
        for each_sentence in test_corpus:
            words_num = words_num + len(each_sentence)
            prob_found = self.getSentenceProbability(each_sentence)
            if(prob_found != 0):
                perplex = perplex + math.log(prob_found,2)
        
        perplexity = 2**((-1/words_num)*perplex) 
        return perplexity
    #endddef
#endclass



# Smoothed bigram language model (use absolute discounting and kneser-ney for smoothing)
class SmoothedBigramModelKN(LanguageModel):
    
    vocab_size = 0
    
    def __init__(self, corpus, vocab, discount,test_corpus):
        
        self.counts = defaultdict(float)
        self.pairCounts = defaultdict(float)
        self.continuation_counts = defaultdict(float)
        self.followed_counts = defaultdict(float)
        self.back_triple_sets = dict()
        self.final_counts = dict()
        self.total = 0.0
        self.final_sum = 0
        self.vocab_size = len(vocab)
        self.discount = discount
        self.train(corpus)
        self.test_corpus = test_corpus
        
    def train(self, corpus):
        for sen in corpus:
            for idx, word in enumerate(sen):
                self.counts[word] += 1.0
                self.total += 1.0
                if(idx > 0):
                    self.pairCounts[(sen[idx-1],word)] += 1
                    if(idx < len(sen)-1):
                        if(word in self.back_triple_sets):
                            seta = self.back_triple_sets[word][0]
                            setb = self.back_triple_sets[word][1]
                            seta.add(sen[idx-1])
                            setb.add(sen[idx+1])
                            self.back_triple_sets[word] = (seta,setb)
                        else:
                            a = set()
                            b = set()
                            self.back_triple_sets[word] = (a,b)
                            
                    elif(word == end):
                        if(word in  self.back_triple_sets):
                            seta = self.back_triple_sets[word][0]
                            setb = self.back_triple_sets[word][0]
                            seta.add(sen[idx - 1])
                            self.back_triple_sets[word] = (seta,setb)
                        else:
                            seta = set()
                            seta.add(sen[idx - 1])
                            setb = set()
                            self.back_triple_sets[word] = (seta,setb)
                             
                            
                elif (word == start):
                    
                    if(word in  self.back_triple_sets):
                        seta = self.back_triple_sets[word][0]
                        setb = self.back_triple_sets[word][1]
                        setb.add(sen[1])
                        self.back_triple_sets[word] = (seta,setb)
                        
                    else:
                         a = set()
                         b = set()
                         b.add(sen[1])
                         self.back_triple_sets[word] = (a,b)
                    
            #endfor
        #endfor
        for key in self.back_triple_sets:
            val = self.back_triple_sets[key]
            seta = val[0]
            setb = val[1]
            lena = len(seta)
            lenb = len(setb)
            self.final_sum = self.final_sum + lenb
            self.final_counts[key] = (lena,lenb)
    #enddef

    def prob(self, word_pair):
         word0, word1 = word_pair
         tuple_found = (word0,word1)
         l_w_c = self.final_counts[word0][0]
         if(l_w_c == 0):
             l_w_c = 1
         p_c = self.final_counts[word1][1] / self.final_sum
         probability = (max(self.pairCounts[tuple_found] - self.discount,0)/self.counts[word0]) + ((self.discount/self.counts[word0])*l_w_c)*(p_c)
         return probability
      
    def getSentenceProbability(self, sen):
        
         prob_found = 0
         is_first = True
         for idx, word in enumerate(sen):
             if(idx != 0):
                 if(is_first):
                     prob_found = self.prob((sen[idx-1],word))
                     is_first = False
                 else:
                     prob_found = prob_found * self.prob((sen[idx-1],word))
         print(prob_found)
         return prob_found
    
    
    def generateSentence(self):
        
        init_word = start
        
        final_sentence = []
        
        final_sentence.append(init_word)
        
        while(init_word != end):
            #rand = random.randint(0,len(self.counts))
            next_options = self.back_triple_sets[init_word][1]
            selected_option = random.choice(list(next_options))
            #print("here")
           # print(selected_option)
            #print("next_option", selected_option)
            init_word = selected_option
            final_sentence.append(init_word)
        
       # print(final_sentence)
        return final_sentence
    
      
    
        
    def getCorpusPerplexity(self, test_corpus):
        words_num = 0
        perplex = 0
        for each_sentence in test_corpus:
            words_num = words_num + len(each_sentence)
            prob_found = self.getSentenceProbability(each_sentence)
            if(prob_found != 0):
                perplex = perplex + math.log(prob_found,2)
        
        perplexity = 2**((-1/words_num)*perplex) 
        return perplexity
    
    
    #endddef
#endclass



# Sample class for a unsmoothed unigram probability distribution
# Note: 
#       Feel free to use/re-use/modify this class as necessary for your 
#       own code (e.g. converting to log probabilities after training). 
#       This class is intended to help you get started
#       with your implementation of the language models above.
class UnigramDist:
    def __init__(self, corpus):
        self.counts = defaultdict(float)
        self.total = 0.0
        self.train(corpus)
    #endddef

    # Add observed counts from corpus to the distribution
    def train(self, corpus):
        for sen in corpus:
            for word in sen:
                if word == start:
                    continue
                self.counts[word] += 1.0
                self.total += 1.0
            #endfor
        #endfor
    #enddef

    # Returns the probability of word in the distribution
    def prob(self, word):
        return self.counts[word]/self.total
    #enddef

    # Generate a single random word according to the distribution
    def draw(self):
        rand = random.random()
        for word in self.counts.keys():
            rand -= self.prob(word)
            if rand <= 0.0:
                return word
	    #endif
	#endfor
    #enddef
#endclass

#-------------------------------------------
# The main routine
#-------------------------------------------

#if __name__ == "__main__":
#    #read your corpora
#    trainCorpus = readFileToCorpus('train.txt')
#    trainCorpus = preprocess(trainCorpus)
#    
#    posTestCorpus = readFileToCorpus('pos_test.txt')
#    negTestCorpus = readFileToCorpus('neg_test.txt')
#    
#    vocab = set()
#    # Please write the code to create the vocab over here before the function preprocessTest
#    print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")
#
#
#    posTestCorpus = preprocessTest(vocab, posTestCorpus)
#    negTestCorpus = preprocessTest(vocab, negTestCorpus)
#
#    # Run sample unigram dist code
#    unigramDist = UnigramDist(trainCorpus)
#    print("Sample UnigramDist output:")
#    print("Probability of \"picture\": ", unigramDist.prob("picture"))
#    print("Probability of \""+UNK+"\": ", unigramDist.prob(UNK))
#    print("\"Random\" draw: ", unigramDist.draw())
#

def get_vocab(corpus):
    corp_set = set()
    for sen in corpus:
        for word in sen:
            if word not in corp_set:
                corp_set.add(word)
                
    return corp_set      




 #read your corpora
trainCorpus = readFileToCorpus('train.txt')
trainCorpus = preprocess(trainCorpus)

posTestCorpus = readFileToCorpus('pos_test.txt')
negTestCorpus = readFileToCorpus('neg_test.txt')

vocab = set()
vocab = get_vocab(trainCorpus)
# Please write the code to create the vocab over here before the function preprocessTest
print("""Task 0: create a vocabulary(collection of word types) for the train corpus""")


posTestCorpus = preprocessTest(vocab, posTestCorpus)
negTestCorpus = preprocessTest(vocab, negTestCorpus)


## Run sample unigram dist code
#unigramDist = UnigramDist(trainCorpus,posTestCorpus)
#print("Sample UnigramDist output:")
#print("Probability of \"picture\": ", unigramDist.prob("picture"))
#print("Probability of \""+UNK+"\": ", unigramDist.prob(UNK))
#print("\"Random\" draw: ", unigramDist.draw())

# Unigram Model

unigramModel = UnigramModel(trainCorpus,posTestCorpus)
unigramModel.generateSentencesToFile(20,"unigram output.txt")
print(unigramModel.getCorpusPerplexity(negTestCorpus))
print(unigramModel.getCorpusPerplexity(posTestCorpus))

# Smoothed Unigram Model

smoothedUnigram = SmoothedUnigramModel(trainCorpus,vocab,posTestCorpus)
smoothedUnigram.generateSentencesToFile(20,"smooth unigram output.txt")
print(smoothedUnigram.getCorpusPerplexity(negTestCorpus))
print(smoothedUnigram.getCorpusPerplexity(posTestCorpus))


# Bigram Model 

bigramModel = BigramModel(trainCorpus, vocab,posTestCorpus)
bigramModel.generateSentencesToFile(20,"bigram output.txt")
print(bigramModel.getCorpusPerplexity(negTestCorpus))
print(bigramModel.getCorpusPerplexity(posTestCorpus))

# Bigram KN

bgmKN = SmoothedBigramModelKN(trainCorpus,vocab,0.5,posTestCorpus)
bgmKN.generateSentencesToFile(20,"smooth bigram kn output.txt")
print(bgmKN .getCorpusPerplexity(negTestCorpus))
print(bgmKN .getCorpusPerplexity(posTestCorpus))

# Very minor difference in perplexity for 2 domains when compared with same model

"""
#Questions

#1 In generating sentences for unigram model the length depends on randomly selecting a word and then adding it to the sentence till end is not reached
For Bigram the length of the sentence is dependent on selecting a word randomly that occurs after the current word in the corpus and thus
proceeding till end is not seen

#2
The models assign drastically different probabilities because for N gram we dont take context and just weigh probabilities on basis of occurence
For Bigram there are lots of 0 bigram which effects the result,
For Bigram with Kneser Nay smoothing the results are better as some weight is assinged to bigrams with 0 occurrence.

#3
Smoothed Bigram because because it assigns some probability to some bigrams initially 0 which increases the number of possible words in sentences.
It also fixes the sparse nature of simple bigram

#4
posTest has the higher perplexity
 
"""