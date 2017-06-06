from sklearn.feature_extraction.text import CountVectorizer
import sys, time


from sklearn import linear_model, datasets
from math import *
import numpy as np;
import warnings
import matplotlib.pyplot as plt
import random;
import os

warnings.filterwarnings("ignore")


class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        print s
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('#' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

def uncertainty_sampling_bagofword(filena):

	xlabel = [];
	for i in range(9):
		xlabel.append(float(i)/10.0 + 0.1);
	ylabel = range(9);
	for i in range(9):
		ylabel[i] = 0.0;

	ana = 20;

	bar = ProgressBar(total = 20)
	for t in range(20):
		bar.move();
		bar.log("uncertainty iteration: "+str(t))
		def file_len(fname):
		    with open(fname) as f:
		        for i, l in enumerate(f):
		            pass
		    return i + 1


		random.seed(t+1);
		holdoutnum = 20;
		filelen = file_len(filena) - holdoutnum;

		shufflelist = range(filelen);


		random.shuffle(shufflelist);

		fp = open(filena);
		y = [];
		corpus = [];
		accuracycorpus = [];
		accuracyy = [];
		whethershown = False;

		for i, line in enumerate(fp):
			if i < filelen:
				vec = line.split("\t")
				if vec[0] == "Ice":
					if (vec[2][0] == 'M'):
						y.append(0);
					elif (vec[2][0] == 'I'):
						y.append(1);
					elif (vec[2][0] == 'C'):
						y.append(2);
					corpus.append(vec[3]);
				if vec[0] == "drinking":
					if (vec[2][0] == 'D'):
						y.append(1);
					elif (vec[2][0] == 'A'):
						y.append(0);
					corpus.append(vec[3]);

			else:
				vec = line.split("\t")
				if vec[0] == "Ice":
					if (vec[2][0] == 'M'):
						accuracyy.append(0);
					elif (vec[2][0] == 'I'):
						accuracyy.append(1);
					elif (vec[2][0] == 'C'):
						accuracyy.append(2);
					accuracycorpus.append(vec[3]);
				if vec[0] == "drinking":
					if (vec[2][0] == 'D'):
						accuracyy.append(1);
					elif (vec[2][0] == 'A'):
						accuracyy.append(0);
					accuracycorpus.append(vec[3]);

		corpus_train = [];
		y_train = [];
		delist = [];

		

		for i in range(filelen / 10):
			corpus_train.append(corpus[shufflelist[i]]);
			y_train.append(y[shufflelist[i]]);






		findone = False;
		findzero = False;
		for i in range(len(y_train)):
			if y_train[i] == 1:
				findone = True;
			else:
				findzero = True;

		if ((findone == False and findzero == True) or (findone == True and findzero == False)):
			ana -= 1;
			continue;

		for i in range(filelen/10):
			delist.append(shufflelist[i]);
		delist.sort();

		for i in range(filelen/10):
			corpus.remove(corpus_train[i]);
			# y.remove(y_train[i]);
			del y[delist[filelen/10-i-1]];



		bigram_vectorizer = CountVectorizer(ngram_range=(1,3), token_pattern=r'\b\w+\b', min_df=1);

		for j in range(9):
			X = bigram_vectorizer.fit_transform(corpus_train).toarray();
			X_test = bigram_vectorizer.transform(corpus).toarray();
			X_acc = bigram_vectorizer.transform(accuracycorpus).toarray()
			lr = linear_model.LogisticRegression()
			lr.fit(X, y_train)
			probalisth = lr.predict_proba(X_test);


			probalist = range(len(probalisth));
			for i in range(len(probalist)):
				tmp = sum([-p*log(p) for p in probalisth[i]]);
				probalist[i] = tmp;
			ylabel[j] += lr.score(X_acc, accuracyy);
			sortlist = np.argsort(probalist);
			sortlist = list(reversed(sortlist));
			for i in range(filelen / 10):
				corpus_train.append(corpus[sortlist[i]]);
				y_train.append(y[sortlist[i]]);

			tmplist = [];
			for i in range(filelen / 10):
				tmplist.append(sortlist[i]);

			tmplist.sort();

			for i in range(filelen / 10):
				corpus.remove(corpus_train[len(corpus_train)-i-1]);
				# y.remove(y_train[len(y_train)-i-1]);
				del y[tmplist[filelen/10 - i - 1]];

			
	for i in range(len(ylabel)):
		ylabel[i] /= float(ana);

	# plt.plot(xlabel, ylabel, '-o');

	# plt.xlim(0, 1)
	# plt.show();
	return (xlabel, ylabel);




def ramdom_sampling_bagofword(filena):

	xlabel = [];
	for i in range(9):
		xlabel.append(float(i)/10.0 + 0.1);
	ylabel = range(9);
	for i in range(9):
		ylabel[i] = 0.0;

	ava = 20;
	bar = ProgressBar(total = 20)

	for j in range(20):
		bar.move();
		bar.log("uncertainty iteration: "+str(j))
		def file_len(fname):
		    with open(fname) as f:
		        for i, l in enumerate(f):
		            pass
		    return i + 1


		random.seed(j+30)
		holdoutnum = 20;

		fp = open(filena);

		filelen = file_len(filena)-holdoutnum
		y = [];
		corpus = [];

		realy = range(filelen)
		realcorpus = range(filelen);
		testy = range(holdoutnum);
		testcorpus = range(holdoutnum);

		corpusfortrain = [];
		corpusfortest = [];


		shufflelist = [];
		for i in range(filelen+holdoutnum):
			shufflelist.append(i);

		random.shuffle(shufflelist);

		bigram_vectorizer = CountVectorizer(ngram_range=(1,3), token_pattern=r'\b\w+\b', min_df=1);


		truechar = '';
		whethershown = False;


		for i, line in enumerate(fp):
			vec = line.split("\t")
			if vec[0] == "Ice":
				if (vec[2][0] == 'M'):
					y.append(0);
				elif (vec[2][0] == 'I'):
					y.append(1);
				elif (vec[2][0] == 'C'):
					y.append(2);
				corpus.append(vec[3]);
			if vec[0] == "drinking":
				if (vec[2][0] == 'D'):
					y.append(1);
				elif (vec[2][0] == 'A'):
					y.append(0);
				corpus.append(vec[3]);

		for i in range(filelen):
			realy[i] = y[shufflelist[i]];
			realcorpus[i] = corpus[shufflelist[i]];

		for i in range(holdoutnum):
			testy[i] = y[shufflelist[i+filelen]];
			testcorpus[i] = corpus[shufflelist[i+filelen]];



		y = realy;
		corpus = realcorpus


		findone = False;
		findzero = False;
		for i in range(filelen/10):
			if y[i] == 1:
				findone = True;
			else:
				findzero = True;


		if ((findone == False and findzero == True) or (findone == True and findzero == False)):
			ava -= 1;
			continue;

		for i in range(9):
			X = bigram_vectorizer.fit_transform(corpus[0:int((i+1)*(float(filelen)/10.0))]).toarray();
			lr = linear_model.LogisticRegression()
			y1 = y[0:int((i+1)*(float(filelen)/10.0))];
			lr.fit(X, y1)
			X = bigram_vectorizer.transform(testcorpus).toarray();
			ylabel[i]+=float(lr.score(X, testy));
			#print(str(int((i+1)*(float(filelen)/10.0))) + " "+str(ylabel[i]) + " "+str(len(X[0])))

	for i in range(9):
		ylabel[i] /= float(ava);
		#print(ylabel[i])

	# plt.plot(xlabel, ylabel, '-o');

	# plt.xlim(0, 1)
	# plt.show();
	return (xlabel, ylabel);
(x3, y3) = uncertainty_sampling_bagofword("ice_train.txt")
(x4, y4) = uncertainty_sampling_bagofword("drinking_train.txt")
(x1,y1) = ramdom_sampling_bagofword("ice_train.txt")
(x2, y2) = ramdom_sampling_bagofword("drinking_train.txt")



plt.clf()

plt.plot(x1, y1, '-o', label = 'ice_ramdom');
plt.plot(x2, y2, '-o', label = 'drinking_ramdom');
plt.plot(x3, y3, '-o', label = 'ice_uncertainty');
plt.plot(x4, y4, '-o', label = 'drinking_uncertainty');

plt.ylim(0.5, 1.01);

plt.legend(bbox_to_anchor=(0.9, 0.5));
plt.show();










