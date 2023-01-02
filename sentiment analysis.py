from textblob import TextBlob
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

import pandas as pd
pd.set_option("display.max_columns", 50)

data = pd.read_csv("train.tsv",sep="\t")

data.head()

data["Sentiment"].replace(0, value="negative", inplace=True)
data["Sentiment"].replace(1, value="negative", inplace=True)
data["Sentiment"].replace(2, value="positive", inplace=True)
data["Sentiment"].replace(3, value="positive", inplace=True)


data = data[(data.Sentiment== "negative") | (data.Sentiment== "positive")]

data.groupby("Sentiment").count()

df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]

# text preprocessing

#upper and lower 
df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
#characters 
df['text'] = df['text'].str.replace('[^\w\s]','')
#digits
df['text'] = df['text'].str.replace('\d','')
#stopwords
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
sw = stopwords.words('english')
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
#to delete rare words 
sil = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]
df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sil))
#lemmi
from textblob import Word
#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

#FEATURE ENGİNEERİNG

#count vectors
#tf-idf vectors (words, characters, n-grams)
df.head()

df.iloc[0] 

#* Count Vectors
#* TF-IDF Vectors (words, characters, n-grams)
#* Word Embeddings

#TF(t) = (Bir t teriminin bir dökümanda gözlenme frekansı) / (dökümandaki toplam terim sayısı) 

#IDF(t) = log_e(Total document count / number of documents with t term in it)

#test- train

x_train, y_train, x_test, y_test = model_selection.train_test_split(df["text"],df["label"])
#>>> x_train.shape 
#(110140,)
#>>> y_train.shape
#(36714,)
#>>> x_test.shape
#(110140,)
#>>> y_test.shape
#(36714,)

encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

#Word Level

#COUNT VECTORS
vectorizer = CountVectorizer()
vectorizer.fit(x_train)
x_train_count = vectorizer.transform(x_train)
x_test_count = vectorizer.transform(x_test)

vectorizer.get_feature_names()[0:5]

x_train_count.toarray() # May stop working based on your ram size



#TF-IDF

tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(x_train)

x_train_tf_idf_word = tf_idf_word_vectorizer.transform(x_train)
x_test_tf_idf_word= tf_idf_word_vectorizer.transform(x_test)

tf_idf_word_vectorizer.get_feature_names()[:5]


#NGRAM LEVEL TF-IDF

tf_idf_ngram_vectorizers = TfidfVectorizer(ngram_range = (2,3))
tf_idf_ngram_vectorizers.fit(x_train)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizers.transform(x_train)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizers.transform(x_test)

# CHARACTER LEVEL TF-IDF

tf_idf_char_vectorizers = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))
tf_idf_char_vectorizers.fit(x_train)

x_train_tf_idf_char = tf_idf_char_vectorizers.transform(x_train)
x_test_tf_idf_char= tf_idf_char_vectorizers.transform(x_test)



################################################################
# MACHINE LEARNING WITH SENTIMENT ANALYSIS
################################################################

######################
# Logistic Regression
######################

#Count Vectorizer
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_count, y_train)
accuracy = model_selection.cross_val_score(loj_model,x_test_count,y_test,cv = 10).mean()
print("Count Vectors Accuracy: ",accuracy)

#TF-IDF Vectorizer
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(loj_model,x_test_tf_idf_word,y_test,cv = 10).mean()
print("Word Level TF-IDF Accuracy: ",accuracy)

#NGRAM Score
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_ngram, y_train)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_ngram, y_test, cv = 10).mean()
print("NGRAM Score:",accuracy)

#Character Level TF-IDF
loj = linear_model.LogisticRegression()
loj_model = loj.fit(x_train_tf_idf_char, y_train)
accuracy = model_selection.cross_val_score(loj_model, x_test_tf_idf_char, y_test, cv=10).mean()
print("Character Level TF-IDF Accuracy: ",accuracy)


######################
#NAIVE BAYES
######################

#Count Vectorizer
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_count, y_train)
accuracy = model_selection.cross_val_score(nb_model, x_test_count, y_test, cv=10).mean()
print("Count Vectors Accuracy: ", accuracy)

#TF-IDF Vectorizer
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_word, y_test, cv=10).mean()
print("Word Level TF-IDF Accuracy: ", accuracy)

#NGRAM Score
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_ngram, y_train)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_ngram, y_test, cv=10).mean()
print("NGRAM Score:", accuracy)

#Character Level TF-IDF
nb = naive_bayes.MultinomialNB()
nb_model = nb.fit(x_train_tf_idf_char, y_train)
accuracy = model_selection.cross_val_score(nb_model, x_test_tf_idf_char, y_test, cv=10).mean()
print("Character Level TF-IDF Accuracy: ", accuracy)


######################
#Random Forest
######################

rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_count,y_train)
accuracy = model_selection.cross_val_score(rf_model, 
                                           x_test_count, 
                                           y_test, 
                                           cv = 10).mean()

print("Count Vectors Accuracy:", accuracy)

#TF-IDF Vectorizer
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_word,y_train)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_word, y_test, cv = 10).mean()
print("Accuracy:", accuracy)

#NGRAM SCORE
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_ngram, y_train)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_ngram, y_test, cv = 10).mean()
print("NGRAM SCORE Accuracy:", accuracy)

#Character Level TF-IDF
rf = ensemble.RandomForestClassifier()
rf_model = rf.fit(x_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(rf_model, x_test_tf_idf_char, y_test, cv=10).mean()
print("Character Level TF-IDF Accuracy: ", accuracy)

######################
#XGBoost
######################

#Count Vectorizers
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_count, y_train)
accuracy = model_selection.cross_val_score(xgb_model, x_test_count, y_test, cv = 10).mean()
print("Count Vectors Accuracy:", accuracy)

#TF-IDF Vectorizers
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_word, y_test, cv = 10).mean()
print("TF-IDF Level Accuracy:", accuracy)

#NGRAM Score
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_ngram, y_train)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_ngram, y_test, cv = 10).mean()
print("NGRAM Score:", accuracy)

#Character Level TF-IDF
xgb = xgboost.XGBClassifier()
xgb_model = xgb.fit(x_train_tf_idf_word, y_train)
accuracy = model_selection.cross_val_score(xgb_model, x_test_tf_idf_char, y_test, cv = 10).mean()
print("Character Level TF-IDF Accuracy: ", accuracy)

######################
# Predictions
######################

pos_comment = pd.Series("this film is very nice and good i like it")
v = CountVectorizer()
v.fit(x_train)
pos_comment = v.transform(pos_comment)
loj_model.predict(pos_comment)
# array ([1])  = positive comment

neg_comment = pd.Series("no not good look at that shit very bad!")
v = CountVectorizer()
v.fit(x_train)
neg_comment = v.transform(neg_comment)
loj_model.predict(neg_comment)
# array ([0])  = negative comment
