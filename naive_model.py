from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_selection import SelectKBest, chi2


def text_reader(text_f_name):
    df = pd.read_csv(text_f_name, sep="\|\|", engine='python')
    return np.asarray(df.values).reshape(-1, )


def label_reader(label_f_name):
    df = pd.read_csv(label_f_name)[['Class']]
    return np.asarray(df.values).reshape(-1, )


def text_analysis_model(train_X, train_y):
    text_clf = Pipeline([
        ('vect', stemmed_count_vect),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB())
    ])
    return text_clf.fit(train_X, train_y)


def write_to_submission_file(prediction):
    # header = 'ID, class1, class2, class3, class4, class5, class6, class7, class8, class9'
    with open('kaggle_submit.csv', 'w') as output_file:
        output_file.write('ID,class1,class2,class3,class4,class5,class6,class7,class8,class9\n')
        n_pred = len(prediction)
        for i in range(n_pred):
            array = [i] + [1 if j == prediction[i] else 0 for j in range(1,10)]
            string = '%s\n' % ','.join([str(k) for k in array])
            output_file.write(string)


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


if __name__ == '__main__':
    # nltk.download("snowball_data")
    # nltk.download('stopwords')
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_count_vect = StemmedCountVectorizer(stop_words='english', ngram_range=(1,2))
    train_X_fname = 'data_sample/training_text'
    train_y_fname = 'data_sample/training_variants'
    test_X_fname = 'data_sample/test_text'
    training_ratio = 0.7
    train_X = text_reader(train_X_fname)
    train_y = label_reader(train_y_fname)
    test_X = text_reader(test_X_fname)
    tfidf = TfidfTransformer()
    _index = int(len(train_X) * training_ratio)
    print 'brute force algorithm ...'
    train_to_valid_X, valid_X = train_X[:_index], train_X[_index:]
    train_to_valid_y, valid_y = train_y[:_index], train_y[_index:]
    best_k = None
    train_acc = None
    test_acc = None
    range_to_brute_force = [100 * i for i in range(1, 101)]
    tran_train_to_valid_X = tfidf.fit_transform(stemmed_count_vect.fit_transform(train_to_valid_X))
    tran_valid_X = tfidf.transform(stemmed_count_vect.transform(valid_X))
    for value in range_to_brute_force:
        k_best = SelectKBest(chi2, k=value)
        train_to_valid_X_new = k_best.fit_transform(tran_train_to_valid_X, train_to_valid_y)
        valid_X_new = k_best.transform(tran_valid_X)
        model = MultinomialNB().fit(train_to_valid_X_new, train_to_valid_y)
        model_train_acc = model.score(train_to_valid_X_new, train_to_valid_y)
        model_testing_acc = model.score(valid_X_new, valid_y)
        if best_k is None:
            best_k = value
            train_acc = model_train_acc
            test_acc = model_testing_acc
            print 'k_best = %s , train_acc = %s, test_acc = %s' % (best_k, train_acc, test_acc)
        elif model_train_acc >= train_acc and model_testing_acc >= test_acc:
            best_k = value
            train_acc = model_train_acc
            test_acc = model_testing_acc
            print 'k_best = %s , train_acc = %s, test_acc = %s' % (best_k, train_acc, test_acc)
        else:
            print 'no improvement'
    print 'best k is founded, best_k = %s' % best_k
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
    tfidf = TfidfTransformer()
    print 'Variable selection ...'
    tran_train_X = tfidf.fit_transform(stemmed_count_vect.fit_transform(train_X))
    print tran_train_X.shape
    k_best = SelectKBest(chi2, k=best_k)
    train_X_new = k_best.fit_transform(tran_train_X, train_y)
    print train_X_new.shape
    print 'Training ... '
    # model = text_analysis_model(train_X, train_y)
    model = MultinomialNB().fit(train_X_new, train_y)
    print 'Predicting ... '
    tran_test_X = tfidf.transform(stemmed_count_vect.transform(test_X))
    test_X_new = k_best.transform(tran_test_X)
    prediction = model.predict(test_X_new)
    # print prediction
    # if 0 in prediction:
    #     print 'contain zero'
    write_to_submission_file(prediction)
