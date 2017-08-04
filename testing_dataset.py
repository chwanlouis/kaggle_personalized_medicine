from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


if __name__ == '__main__':
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                                      categories=categories,
                                      shuffle=True,
                                      random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    # print X_train_counts.shape
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    # print X_train_tf.shape
    clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
    docs_new = list()
    with open('data_sample/test_data') as f:
        while True:
            _string = f.readline().replace('\n', '')
            if len(_string) > 0:
                docs_new.append(_string)
            else:
                break
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, predicted):
        print '%s => %s' % (doc, twenty_train.target_names[category])