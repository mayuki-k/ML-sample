from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from env import make5
import util

x_train, x_test, y_train, y_test = make5.get_datas()

vectorizer = CountVectorizer(stop_words = 'english')
vectorizer.fit(x_train)

x_train_bow = vectorizer.transform(x_train)
x_test_bow = vectorizer.transform(x_test)

def naive_bayes(alpha, x_bow, y):
    f = MultinomialNB(alpha=alpha)
    f.fit(x_bow, y)
    return f

f = naive_bayes(0.4, x_train_bow, y_train)
util.show_score(f, x_test_bow, y_test, 'Test')