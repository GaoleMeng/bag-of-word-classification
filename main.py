from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, datasets
import numpy as np;
import warnings
import matplotlib.pyplot as plt
import random;
warnings.filterwarnings("ignore")


random.seed(1)
lr = linear_model.LogisticRegression()
fp = open("drinking_train.txt");
y = [];
corpus = [];

realy = range(100)
realcorpus = range(100);

corpusfortrain = [];
corpusfortest = [];
xlabel = [];
ylabel = [];
shufflelist = [];
for i in range(100):
	shufflelist.append(i);

random.shuffle(shufflelist);

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1);

for i in range(9):
	xlabel.append(float(float(i)/10.0) + 0.1);

# plt.plot(xlabel, ylabel, '-');
# 

for i, line in enumerate(fp):
	vec = line.split("\t")
	if (vec[2][0] == 'D'):
		y.append(0);
	else:
		y.append(1);
	corpus.append(vec[3]);

for i in range(100):
	realy[i] = y[shufflelist[i]];
	realcorpus[i] = corpus[shufflelist[i]];

y = realy;
corpus = realcorpus

print(len(corpus));

X = bigram_vectorizer.fit_transform(corpus).toarray();

for i in range(9):
	X = bigram_vectorizer.fit_transform(corpus[0:(i+1)*10]).toarray();
	lr.fit(X, y[0:(i+1)*10])
	X = bigram_vectorizer.transform(corpus).toarray();
	ylabel.append(lr.score(X,y));

plt.plot(xlabel, ylabel, '-');
plt.show();
