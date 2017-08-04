import pickle
import string


def reading_text_file(file_path):
    all_word = list()
    with open(file_path, 'r') as f:
        header = f.readline()
        while True:
            _string = f.readline().replace('\r', '').replace('\n', '')
            if len(_string) == 0:
                break
            _id, text = _string.split('||')
            word_dictionary, total_word = word_counter(text)
            pickle_dict_f_path = 'pickles/word_dict_id_%s.pickle' % _id
            with open(pickle_dict_f_path, 'wb') as dict_f:
                pickle.dump(word_dictionary, dict_f)
                print 'File %s generated' % pickle_dict_f_path
            for word in word_dictionary.keys():
                if word not in all_word:
                    all_word.append(word)
                    # print 'word "%s" added' % word
    pickle_list_f_path = 'pickles/all_word_list.pickle'
    with open(pickle_list_f_path, 'wb') as list_f:
        pickle.dump(all_word, list_f)
        print 'File %s generated' % pickle_list_f_path
    print 'Total valid word in training set : %s' % len(all_word)


def word_treat(word):
    all_valid_char = string.ascii_letters + string.digits + ['*']
    if len(word) == 0:
        return None
    while word[0] not in all_valid_char:
        word = word[1:]
        if len(word) == 0:
            return None
    while word[-1] not in all_valid_char:
        word = word[:-1]
    return word


def word_counter(full_text, delimiter=' '):
    array = full_text.split(delimiter)
    word_dict = dict()
    total_word = 0
    for word in array:
        word = word_treat(word)
        if word is None or not word.isalnum():
            continue
        if word not in word_dict.keys():
            word_dict[word] = 1
        else:
            word_dict[word] += 1
        total_word += 1
    return word_dict, total_word


if __name__ == '__main__':
    training_f_path = 'data_sample/training_text'
    read_training_file = False
    if read_training_file:
        reading_text_file(training_f_path)
