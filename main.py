from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, datasets
import numpy as np;
import warnings
import matplotlib.pyplot as plt
import random;
import os

warnings.filterwarnings("ignore")


def uncertainty_sampling_bagofword(filena):

	def file_len(fname):
	    with open(fname) as f:
	        for i, l in enumerate(f):
	            pass
	    return i + 1

	xlabel = [];
	for i in range(9):
		xlabel.append(float(i)/10.0 + 0.1);
	ylabel = range(9);
	for i in range(9):
		ylabel[i] = 0.0;

	random.seed(1);

	filelen = file_len(filena);

	shufflelist = range(filelen);

	random.shuffle(shufflelist);

	y = [];
	corpus = [];

	for i, line in enumerate(fp):
		vec = line.split("\t")
		if (whethershown == False):
			truechar = vec[3][0];
			whethershown = True;
			y.append(1);
		elif (whethershown == True):
			if (truechar == vec[3][0]):
				y.append(1);
			else:
				y.append(0);
		corpus.append(vec[3]); 


	corpus_train = [];
	y_train = [];
	for i in range(filelen / 10);
		corpus_train.append(corpus[shufflelist[i]]);
		y_train.append(y[shufflelist[i]]);

	for i in range(filelen/10):
		corpus.remove(corpus_train[i]);


	X = bigram_vectorizer.fit_transform(corpus_train).toarray();
	lr = linear_model.LogisticRegression()
	lr.fit(X, y_train)
	X = bigram_vectorizer.transform(testcorpus).toarray();



	# for i in range()



def ramdom_sampling_bagofword(filena):

	xlabel = [];
	for i in range(9):
		xlabel.append(float(i)/10.0 + 0.1);
	ylabel = range(9);
	for i in range(9):
		ylabel[i] = 0.0;

	ava = 100;

	for j in range(100):
		def file_len(fname):
		    with open(fname) as f:
		        for i, l in enumerate(f):
		            pass
		    return i + 1


		random.seed(j+30)
		holdoutnum = 60;

		fp = open(filena);

		filelen = file_len(filena)-holdoutnum
		print(filelen)
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
			if (whethershown == False):
				truechar = vec[3][0];
				whethershown = True;
				y.append(1);
			elif (whethershown == True):
				if (truechar == vec[3][0]):
					y.append(1);
				else:
					y.append(0);
			corpus.append(vec[3]); 

		for i in range(filelen):
			realy[i] = y[shufflelist[i]];
			realcorpus[i] = corpus[shufflelist[i]];

		for i in range(holdoutnum):
			testy[i] = y[shufflelist[i+filelen]];
			testcorpus[i] = corpus[shufflelist[i+filelen]];

		print(testy);


		y = realy;
		corpus = realcorpus


		findone = False;
		findzero = False;
		for i in range(6):
			if y[i] == 1:
				findone = True;
			else:
				findzero = True;


		if ((findone == False and findzero == True) or (findone == True and findzero == False)):
			ava -= 1;
			continue;
		print(y);
		print(len(testy))


		for i in range(9):
			X = bigram_vectorizer.fit_transform(corpus[0:int((i+1)*(float(filelen)/10.0))]).toarray();
			lr = linear_model.LogisticRegression()
			y1 = y[0:int((i+1)*(float(filelen)/10.0))];
			lr.fit(X, y1)
			X = bigram_vectorizer.transform(testcorpus).toarray();
			ylabel[i]+=float(lr.score(X, testy));
			print(str(int((i+1)*(float(filelen)/10.0))) + " "+str(ylabel[i]) + " "+str(len(X[0])))

	for i in range(9):
		ylabel[i] /= float(ava);
		print(ylabel[i])

	plt.plot(xlabel, ylabel, '-o');

	plt.xlim(0, 1)
	plt.show();

# ramdom_sampling_bagofword("ice_train.txt")
