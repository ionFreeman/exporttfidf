import graphlab
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from urllib import quote_plus


'''Check GraphLab Create version'''
from distutils.version import StrictVersion
assert (StrictVersion(graphlab.version) >= StrictVersion('1.8.5')), 'GraphLab Create must be version 1.8.5 or later.'
print np.version.version

abstracts = graphlab.SFrame('/Users/ion/Pace/AbstractScraper/abstractsbypath.csv', format='csv')

abstracts['tf_idf'] = graphlab.text_analytics.tf_idf(abstracts['Abstract'])


def sframe_to_scipy(x, column_name):
    '''
    Convert a dictionary column of an SFrame into a sparse matrix format where
    each (row_id, column_id, value) triple corresponds to the value of
    x[row_id][column_id], where column_id is a key in the dictionary.

    Example
    >>> sparse_matrix, map_key_to_index = sframe_to_scipy(sframe, column_name)
    '''
    assert x[column_name].dtype() == dict, \
        'The chosen column must be dict type, representing sparse data.'

    # Create triples of (row_id, feature_id, count).
    # 1. Add a row number.
    x = x.add_row_number()
    # 2. Stack will transform x to have a row for each unique (row, key) pair.
    x = x.stack(column_name, ['feature', 'value'])

    # Map words into integers using a OneHotEncoder feature transformation.
    f = graphlab.feature_engineering.OneHotEncoder(features=['feature'])
    # 1. Fit the transformer using the above data.
    f.fit(x)
    # 2. The transform takes 'feature' column and adds a new column 'feature_encoding'.
    x = f.transform(x)
    # 3. Get the feature mapping.
    mapping = f['feature_encoding']
    # 4. Get the feature id to use for each key.
    x['feature_id'] = x['encoded_features'].dict_keys().apply(lambda x: x[0])

    # Create numpy arrays that contain the data for the sparse matrix.
    i = np.array(x['id'])
    j = np.array(x['feature_id'])
    v = np.array(x['value'])
    width = x['id'].max() + 1
    height = x['feature_id'].max() + 1

    # Create a sparse matrix.
    mat = csr_matrix((v, (i, j)), shape=(width, height))

    return mat, mapping

# The conversion will take about a minute or two.
tf_idf, map_index_to_word = sframe_to_scipy(abstracts, 'tf_idf')

from sklearn.preprocessing import normalize
tf_idf = normalize(tf_idf)
(rows, cols) =tf_idf.shape
# save the tf idf matrix to a file
# np.savetxt(fname='pacedpsabstracts.csv', X=tf_idf, fmt='%10.5f', delimiter=' ')
with open('abstractswithtfidf.csv', 'w+b', 4096) as output:
    output.write(','.join(['AbstractPath', 'AbstractText'] + [quote_plus("{}".format(map_index_to_word[b]['category'])) for b in xrange(cols)]))
    output.write('\n')
    for a in xrange(rows):
        # urlencode the fields
        fields = map(lambda x:quote_plus(x), map(lambda y: abstracts[y][a], ['Path', 'Abstract']) + ["{}".format(tf_idf[a, b] * 100000.0) for b in xrange(cols)])
        output.write("{}\n".format(','.join(fields)))

