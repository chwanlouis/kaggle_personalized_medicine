import re
import string
import pickle
import pandas as pd


def direct_name_matching(training_variants_df):
    nrow = len(training_variants_df)
    gene_match = 0
    variation_match = 0
    for i in range(nrow):
        _id, _gene, _variation, _class = training_variants_df[i:i+1].values[0]
        file_name = 'pickles/word_dict_id_%s.pickle' % _id
        with open(file_name, 'rb') as f:
            word_dict = pickle.load(f)
            print 'Pickle loaded %s' % (file_name)
        if _gene in word_dict.keys():
            gene_match += 1
        if _variation in word_dict.keys():
            variation_match += 1
    print 'Gene match : %.4f percent' % (100 * float(gene_match) / nrow)
    print 'Variation match : %.4f percent' % (100 * float(variation_match) / nrow)


def get_gene_variation_word(word_list):
    output = list()
    for word in word_list:
        is_append = True
        for char in word:
            if char.islower():
                is_append = False
                break
        if word.isdigit():
            is_append = False
        if len(word) < 3:
            is_append = False
        if is_append and len():
            output.append(word)
    return output


if __name__ == '__main__':
    training_variants_file_name = 'data_sample/training_variants'
    testing_variants_file_name = 'data_sample/test_variants'
    training_var_df = pd.read_csv(training_variants_file_name)
    testing_var_df = pd.read_csv(testing_variants_file_name)
    all_gene_var = training_var_df[['Gene', 'Variation']].values.tolist() + \
        testing_var_df[['Gene', 'Variation']].values.tolist()
    valid_char = string.letters + string.digits
    exceptional_char = list()
    for word in all_gene_var:
        gene, var = word
        for char in gene:
            if char not in valid_char:
                if char not in exceptional_char:
                    exceptional_char.append(char)
        for char in var:
            if char not in valid_char:
                if char not in exceptional_char:
                    exceptional_char.append(char)
    print exceptional_char
    # direct_name_matching(training_var_df)
    '''
    with open('pickles/all_word_list.pickle', 'rb') as f:
        all_words = pickle.load(f)
    print get_gene_variation_word(all_words)
    '''
