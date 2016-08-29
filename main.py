import json
import csv
import math
from collections import OrderedDict
from operator import itemgetter

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    if bowCount == 0:
        return tfDict
    for word, count in wordDict.iteritems():
        tfDict[word] = count / float(bowCount)
        
    return tfDict

def computeIDF(docList):
    idfDict = {}
    N = len(docList)
    #counts the number of documents that contain a word w
    idfDict = dict.fromkeys(docList[0].keys(),0)
    for doc in docList:
        for word, val in doc.iteritems():
            if val > 0:
                idfDict[word] += 1.0
                
    #divide N by denominator above, take the log of that
    for word, val in idfDict.iteritems(): 
        idfDict[word]= math.log(N / float(val)) 

    return idfDict

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.iteritems():
        tfidf[word] = val * idfs[word]       
    return tfidf


def solve():    
#====================Read Data======================

 fname = "TestingDataForTDIDF.json"
 file_data = ""
 with open(fname, 'r') as fin:
  file_data = fin.read()
 data = json.loads(file_data)
 
 with open('stopWords.csv', 'rb') as f:
  reader = csv.reader(f)
  inp2 = list(reader)
  StopWord = inp2[0]
   
 WordsDict = {}
 with open('polarity.csv', 'rb') as f:
  reader = csv.reader(f)
  WordList = list(reader)

 for row in WordList:
  WordsDict[str(row[1])] = (float(row[2]),row[0])

 with open('nouns.csv', 'rb') as f:
  reader = csv.reader(f)
  inp = list(reader)
 Nouns = inp[0]

 negationWords = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

#===============Start of Processing=================

 NounMain = {}
 punc = ",<>?/}{()*&^%$#@!\\"
 
 for i in range(len(data['Reviews'])):
  x = data['Reviews'][i]['Content'].split('.')  
  for sentence in x:    
   arr = sentence.split(' ')
   for cstr in arr:
    if cstr not in punc:   
     p = str(cstr).lower()
     if p in Nouns and p not in StopWord:
      NounMain[p] = 0
     
 WordMain = []
 BowMain = []

# Genearate Bag of Words and Dict
 for i in range(len(data['Reviews'])):
  x = data['Reviews'][i]['Content'].split('.')  
  bow = []
  nounRev = NounMain.copy()
  for sentence in x:    
   arr = sentence.split(' ')   
   for cstr in arr:
    p = str(cstr).lower()
    if p in NounMain and p not in StopWord:
     bow.append(p)
     nounRev[p] += 1
  WordMain.append(nounRev)   
  BowMain.append(bow) 

# calculate TF-IDF
 
 tfBowMain = []

 for index in range(len(WordMain)):
    tfBowMain.append( ( computeTF( WordMain[index], BowMain[index]) ) ) 
 
 idfs = computeIDF(WordMain)
  
 tfidfMain = []
 
 for index in range(len(tfBowMain)):
    tfidfMain.append( ( computeTFIDF(tfBowMain[index], idfs) ) )
 
# Feature Selection via S.D.
 
 Sum = 0
 Count = 0
 itr = 0
 
 d = OrderedDict(sorted(tfidfMain[itr].items(), key=itemgetter(1)))
 
 for i in range(len(tfidfMain)):
  for word, val in tfidfMain[i].iteritems():
    if val > 0:
        Sum += val
        Count += 1
 Avg = Sum/Count

 SD = 0
 
 for i in range(len(tfidfMain)):
  for word, val in tfidfMain[i].iteritems():
     if val > 0:
         SD += (val-Avg)**2
 SD /= Count
 SD = math.sqrt(SD)

 phi = 0
 Features = []
 
# For features we have weights with which will plot Bar Graph, Features Weights Dict formation

 maxlen = 0
 
 for i in range(5,15):
  NewDict = {}   
  for word, val in tfidfMain[i].iteritems():
     if  val > phi:
      Features.append( (word,val) )
      NewDict[word] = val
  d = OrderedDict(sorted(NewDict.items(), key=itemgetter(1),reverse = True))
      
 Features.sort(key=lambda x: -x[1])

 print "\n\n ================ TF-IDF Score for Selected Features ================ \n\n"
 for a, b in Features:
     print "  ", "{0:.6f}".format(b), " : ", a

 
 print "\n\n  ================ Sentiment Analysis Result ================ ", '\n'

 print "Processed Text:"
 print '\n',data['Reviews'][itr]['Content'],'\n\n'
   
# Adjective and Noun Pairing
 FeatureDict = {}
 avg = 0
 for sentence in data['Reviews'][itr]['Content'].split('.'):
     punc = list(",./;'?&-")

     for line in sentence:
        if line in punc:
            sentence = sentence.replace(line,"") 
     #print sentence
     Fptr = []
     AdjPtr = []
     arr = sentence.split(' ')
     i = 0
     direction = +1
     for cstr in arr:
       i += 1
       p = str(cstr).lower()
       if p in negationWords:
           direction = -direction
           continue
       if p == "but":
           direction = +1
           continue
       if p in StopWord:
           continue
       elif p in tfidfMain[itr].keys():
           Fptr.append( (p,i) )
           if p not in FeatureDict.keys():
               FeatureDict[p] = 0.0
               if p in WordsDict.keys():
                   FeatureDict[p] = WordsDict[p][0]*0.1
       elif p in WordsDict.keys():
            AdjPtr.append( (p,direction*WordsDict[p][0],i) )
     # print AdjPtr

     for i in range(len(AdjPtr)):
         dist = 1000000007
         feat = ""
         for j in range(len(Fptr)):
             if abs(AdjPtr[i][2] - Fptr[j][1]) < dist:
                 dist = abs(AdjPtr[i][2] - Fptr[j][1])
                 feat = Fptr[j][0]
         if feat != '':
             FeatureDict[feat] += AdjPtr[i][1]
         avg += AdjPtr[i][1]
         #print feat + " - " + AdjPtr[i][0]

 print "Adjective Score:\n\n", FeatureDict, '\n'
 avg /= max(1,len(FeatureDict))
 print "Review score: ", avg
 
 return 0

print "Exec"
solve()
  
