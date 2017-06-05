from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, datasets
import numpy as np;
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


lr = linear_model.LogisticRegression()
# X = np.random.randn(3, 4)
# y = [1,0,0]
# lr.fit(X,y);
# print(lr.predict_proba(X[0, :]));


fp = open("drinking_train.txt");
y = [];
corpus = []
corpusfortrain = [];
corpusfortest = [];
xlabel = [];
ylabel = [];
shufflelist = [];
for i in range(100):
	shufflelist.append(i);



bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1);

for i in range(9):
	xlabel.append(float(float(i)/10.0) + 0.1);

for i in range(9):
	ylabel.append(0.1);


plt.plot(xlabel, ylabel, '-');
plt.show(); 


for i, line in enumerate(fp):
	vec = line.split("\t")
	if (vec[2][0] == 'D'):
		y.append(0);
	else:
		y.append(1);
	corpus.append(vec[3]);

X = bigram_vectorizer.fit_transform(corpus).toarray();
lr.fit(X,y);

print(len(y));

# y = [];
# corpus = [];

X = bigram_vectorizer.transform(corpus).toarray();
print(len(y), len(X));

for i in range(10):
	print("the answer is " + str(y[i]) + " the prediction is " + str(lr.predict(X[i, :])))


# print(lr.predict_proba(X[0, :]));




# print(y);




# vectorizer = CountVectorizer(min_df = 1);
# corpus = ['This is the first document.', 'This is the second second document.','And the third one.', 'Is this the first document?']


# analyze = vectorizer.build_analyzer()
#  analyze("This is a text document to analyze.")



# X = vectorizer.fit_transform(corpus)

# vectorizer.get_feature_names()

# print(X.toarray());