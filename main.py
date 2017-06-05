from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, datasets
import numpy as np;
import warnings
import matplotlib.pyplot as plt
import random;
import os

warnings.filterwarnings("ignore")

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

random.seed(1)
lr = linear_model.LogisticRegression()
fp = open("ice_train.txt");

filelen = file_len("ice_train.txt")
print(filelen)
y = [];
corpus = [];

realy = range(filelen)
realcorpus = range(filelen);

corpusfortrain = [];
corpusfortest = [];
xlabel = [];
ylabel = [];
shufflelist = [];
for i in range(filelen):
	shufflelist.append(i);

random.shuffle(shufflelist);

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1);

for i in range(9):
	xlabel.append(float(i)/10.0 + 0.1);



# plt.plot(xlabel, ylabel, '-');
# 

truechar = '';
whethershown = False;


for i, line in enumerate(fp):
	vec = line.split("\t")
	if (whethershown == False):
		truechar = vec[3][0];
		whethershown = True;
		y.append(0);
	elif (whethershown == True):
		if (truechar == vec[3][0]):
			y.append(0);
		else:
			y.append(1);
	corpus.append(vec[3]);

for i in range(filelen):
	realy[i] = y[shufflelist[i]];
	realcorpus[i] = corpus[shufflelist[i]];

y = realy;
corpus = realcorpus

print(y);

X = bigram_vectorizer.fit_transform(corpus).toarray();


for i in range(9):
	print(int((i+1)*(float(filelen)/10.0)))
	X = bigram_vectorizer.fit_transform(corpus[0:int((i+1)*(float(filelen)/10.0))-1]).toarray();
	lr.fit(X, y[0:int((i+1)*(float(filelen)/10.0))-1])
	X = bigram_vectorizer.transform(corpus).toarray();
	ylabel.append(lr.score(X,y));

plt.plot(xlabel, ylabel, '-o');
plt.xlim(0, 1)
plt.ylim(0, 1.1)
plt.show();
