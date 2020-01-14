import sys
import os
from _collections import defaultdict
import datetime
import time
from gensim.models import word2vec

def Run(w2vModelFile,outputFile="./output/gstem{}.txt",maxEditDist=2,top_n_similarity=100,way=1,amnt=1,lhs=1):
    _alphabet = "سألتمونيها"
    _weights = defaultdict(lambda: (1,1,1))
    for letter in _alphabet:
        if(letter == "ا" or letter == "ي" or letter == "و"):
            _weights[letter] = (way,way,way)
        elif(letter == "أ" or letter == "م" or letter == "ن" or letter == "ت"):
            _weights[letter] = (amnt,amnt,amnt)
        else :
            _weights[letter] = (lhs,lhs,lhs)
    def iterative_levenshtein(s, t):
        rows = len(s) + 1
        cols = len(t) + 1
        if (rows == 1 or cols == 1):
            return 1000
        dist = [[0 for x in range(cols)] for x in range(rows)]
        for row in range(1, rows):
            dist[row][0] = dist[row - 1][0] + _weights[s[row - 1]][0]
        for col in range(1, cols):
            dist[0][col] = dist[0][col - 1] + _weights[t[col - 1]][1]
        for col in range(1, cols):
            for row in range(1, rows):
                delWeight = 1
                insWeight = 1
                subWeightRow = 1
                subWeightCol = 1
                if(s[row - 1] in _weights):
                    delWeight = _weights[s[row - 1]][0]
                if(t[col - 1] in _weights):
                        insWeight = _weights[t[col - 1]][1]
                if(s[row - 1] in _weights):
                        subWeightRow = _weights[s[row - 1]][2]
                if(t[col - 1] in _weights):
                    subWeightCol = _weights[t[col - 1]][2]
                
                deletes = delWeight
                inserts = insWeight
                subs = max((subWeightRow, subWeightCol))
                if s[row - 1] == t[col - 1]:
                    subs = 0
                else:
                    subs = subs
                dist[row][col] = min(dist[row - 1][col] + deletes,
                                        dist[row][col - 1] + inserts,
                                        dist[row - 1][col - 1] + subs) # substitution
        return dist[row][col]
    def hasDiffNormalLetter(key,otherKey):
         result = False
         for letter in key:
            if(_alphabet.find(letter) == -1 and otherKey.find(letter) == -1):
                result = True
                break
         if(not result):
            for letter in otherKey:
                if(_alphabet.find(letter) == -1 and key.find(letter) == -1):
                    result = True
                    break
         return result
    outputFile = outputFile.format(int(time.time()))
    print("G-Stem file {}".format(outputFile))
    wvModel = word2vec.Word2Vec.load(w2vModelFile, mmap='r')
    keys = list(wvModel.wv.vocab.keys())
    keys.sort(key = len)
    wordMap = []
    ind = 0
    lenAll = len(keys)
    startTime = datetime.datetime.now()
    if os.path.exists(outputFile):
        os.remove(outputFile)
    mappedKeys = []
    mappedVlaues = []
    for key in keys:
        ind = ind + 1
        sys.stdout.write("\r{} of {} words => {:0.00%} time is {} dictionary len {}".format(ind,lenAll,ind / lenAll,datetime.datetime.now() - startTime,len(mappedVlaues)))
        sys.stdout.flush()
        minDist = []
        nearestItems = wvModel.most_similar([key],topn=top_n_similarity)
        for nearstitem in nearestItems :
            otherKey = nearstitem[0]
            StopLoop = False
            if(hasDiffNormalLetter(key,otherKey)):
               continue
            letterDist = iterative_levenshtein(key,otherKey)
            if(letterDist <= maxEditDist):
                minDist.append(otherKey)
        if(len(minDist)):
            with open(outputFile, "a") as output:
                for item in minDist:
                    itemInKeys = item in mappedKeys
                    itemInVlaues = item in mappedVlaues
                    keyInValues = key in mappedVlaues
                    if((itemInKeys and keyInValues) or (itemInVlaues and keyInValues)):
                        break
                    elif(itemInKeys):
                        output.write(item + "," + key + "\n")
                        mappedKeys.append(item)
                        mappedVlaues.append(key)
                    elif(itemInVlaues):
                        idx = mappedVlaues.index(item)
                        orignalKey = mappedKeys[idx]
                        if(orignalKey != key):
                            output.write(orignalKey + "," + key + "\n")
                            mappedKeys.append(orignalKey)
                            mappedVlaues.append(key)
                    elif(keyInValues):
                        idx = mappedVlaues.index(key)
                        orignalKey = mappedKeys[idx]
                        if(orignalKey != item):
                            output.write(orignalKey + "," + item + "\n")
                            mappedKeys.append(orignalKey)
                            mappedVlaues.append(item)
                    else:
                        output.write(key + "," + item + "\n")
                        mappedKeys.append(key)
                        mappedVlaues.append(item)