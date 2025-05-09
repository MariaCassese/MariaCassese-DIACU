import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from oversampling.dro import DistributionalRandomOversampling
import string
from scipy.sparse import hstack, csr_matrix, issparse
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import Normalizer
import numpy as np
from tqdm import tqdm
import pickle
from nltk import ngrams

from string import punctuation

class DocumentProcessor:
    def __init__(self, language_model=None, savecache='.cache/processed_docs_def.pkl'):
        self.nlp = language_model
        self.savecache = savecache
        self.init_cache()

    def init_cache(self):
        if self.savecache is None or not os.path.exists(self.savecache):
            print('Cache not found, initializing from scratch')
            self.cache = {}
        else:
            print(f'Loading cache from {self.savecache}')
            self.cache = pickle.load(open(self.savecache, 'rb'))

    def save_cache(self):
        if self.savecache is not None:
            print(f'Storing cache in {self.savecache}')
            parent = Path(self.savecache).parent
            if parent:
                os.makedirs(parent, exist_ok=True)
            pickle.dump(self.cache, open(self.savecache, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def delete_doc(self, filename):
        removed_doc = self.cache.pop(filename, None)
        if removed_doc is not None:
            print(f'Removed {filename} from cache')
            self.save_cache() 
        
        else:
            print(f'{filename} not found in cache')
            
            
    def process_documents(self, documents, filenames):
        processed_docs = {}
        for filename, doc in zip(filenames, documents):
            if filename in self.cache:
                processed_docs[filename[:-2]] = self.cache[filename]
            else:
                print(f'{filename} not in cache')
                processed_doc = self.nlp(doc)
                self.cache[filename] = processed_doc
                processed_docs[filename[:-2]] = self.cache[filename]
                self.save_cache()
        return processed_docs 


class DummyTfidf:

    def __init__(self,upto, feature_type="word"):
        assert feature_type in {'word', 'sentence'}, 'feature type not valid'
        self.upto = upto
        self.prefix = f"{feature_type}_length" 

    def get_feature_names_out(self):
        return np.array([f"{self.prefix}_{i}" for i in range(1, self.upto)])


class FeaturesMendenhall:
    """
    Extract features as the frequency of the words' lengths used in the documents,
    following the idea behind Mendenhall's Characteristic Curve of Composition
    """
    def __init__(self,upto=25):
        self.upto = upto
        self.vectorizer = DummyTfidf(self.upto)

    def __str__(self) -> str:
        return 'FeaturesMendenhall'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in tqdm(documents, 'Extracting word lenghts', total=len(documents)):
            word_lengths = [len(str(token)) for token in doc]
            hist = np.histogram(word_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distribution = np.cumsum(hist)
            features.append(distribution)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)


class FeaturesSentenceLength:
    def __init__(self, upto=1000, language='russian'):
        self.upto = upto
        self.language = language
        self.vectorizer = DummyTfidf(self.upto)

    def __str__(self) -> str:
        return 'FeaturesSentenceLength'

    def fit(self, documents, y=None):
        return self

    def transform(self, documents, y=None):
        features = []
        for doc in tqdm(documents, 'Extracting sentence lenghts', total=len(documents)):
            sentence_lengths = []
            for sentence in doc.sents:
                sent_len = [len(str(token)) for token in sentence]
                sentence_lengths += sent_len
            hist = np.histogram(sentence_lengths, bins=np.arange(1, self.upto), density=True)[0]
            distributuion = np.cumsum(hist)
            features.append(distributuion)
        return np.asarray(features)

    def fit_transform(self, documents, y=None):
        return self.fit(documents).transform(documents)


class FeaturesCharNGram:

    def __init__(self, n=(1,3), sublinear_tf=False, norm='l1'):
        self.n = n
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.counter = CountVectorizer(analyzer='char', ngram_range=self.n, min_df=3)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3)
    
    def __str__(self) -> str:
        return f'FeaturesCharNGram [n-gram range: ({self.n[0]},{self.n[1]})]'

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_ngrams(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_ngrams(raw_documents)
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(self.n), use_idf=False, norm=self.norm, min_df=3)
        return self.vectorizer.fit_transform(raw_documents)

    
    def count_ngrams(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_ngrams = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_ngrams.sum(axis=1).getA().flatten()
        else:
            self.test_ngrams = self.counter.transform(texts)
            self.n_test_terms = self.test_ngrams.sum(axis=1).getA().flatten()
        

class FeaturesPunctuation:

    def __init__(self, sublinear_tf=False, norm='l1', ngram_range=(1,3)):
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.punctuation=punctuation
        self.ngram_range = ngram_range
        self.counter = CountVectorizer(vocabulary=self.punctuation, min_df=1)
        self.vectorizer = TfidfVectorizer(analyzer='char', vocabulary=self.punctuation, use_idf=False, norm=self.norm, min_df=3, ngram_range=self.ngram_range)
    
    def __str__(self) -> str:
        ngram_range_str = f' [n-gram range: {self.ngram_range}]'
        return 'FeaturesPunctuation' + ngram_range_str

    def fit(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.vectorizer.fit(raw_documents)
        return self

    def transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.transform(raw_documents)

    def fit_transform(self, documents, y=None):
        raw_documents = [doc.text for doc in documents]
        self.count_words(raw_documents)
        return self.vectorizer.fit_transform(raw_documents)

    def count_words(self, texts):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(texts)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(texts)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()


class FeaturesPOST:
    def __init__(self, n=(1,4), use_idf=True, sublinear_tf=True, norm='l2', savecache='.postcache/dict.pkl', **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        self.n = n
        self.counter = CountVectorizer(analyzer=self.post_analyzer)
        self.vectorizer = TfidfVectorizer(analyzer=self.post_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
    

    def __str__(self) -> str:
        return f'FeaturesPOST [n-gram range: ({self.n[0]},{self.n[1]})]'


    def post_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (self.n)) # up to quadrigrams
        ngram_range = slice(*ngram_range)
        ngram_tags = []
        
        for sentence in doc.sents:
            sentence_unigram_tags = [token.pos_ if token.pos_ != '' else 'Unk' for token in sentence]
            for n in list(range(ngram_range.start, ngram_range.stop+1)):
                sentence_ngram_tags = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_tags, n))]
                ngram_tags.extend(sentence_ngram_tags)
        return ngram_tags


    def fit(self, documents, y=None):
        self.count_pos_tags(documents)
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents, y=None):
        self.count_pos_tags(documents)
        post_features = self.vectorizer.transform(documents)
        row0 = post_features[0]
        nz_cols = row0.nonzero()[1]
        feature_names = self.vectorizer.get_feature_names_out()
        return post_features

    def fit_transform(self, documents, y=None):
        self.count_pos_tags(documents)
        post_features = self.vectorizer.fit_transform(documents)
        row0 = post_features[0]
        nz = row0.nonzero()[1]
        return post_features

    def count_pos_tags(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()
    

class FeaturesDEP:
    def __init__(self, n=(1,3), use_idf=True, sublinear_tf=True, norm='l2', savecache='.depcache/dict.pkl', **tfidf_kwargs):
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.tfidf_kwargs = tfidf_kwargs
        self.savecache = savecache
        self.n = n
        self.counter = CountVectorizer(analyzer=self.dep_analyzer)
        self.vectorizer = TfidfVectorizer(analyzer=self.dep_analyzer, use_idf=self.use_idf, sublinear_tf=self.sublinear_tf, norm=self.norm, **self.tfidf_kwargs)
    
    def __str__(self) -> str:
        return f'FeaturesDEP [n-gram range: ({self.n[0]},{self.n[1]})]'
    

    def dep_analyzer(self, doc):
        ngram_range = self.tfidf_kwargs.get('ngram_range', (self.n))
        ngram_range = slice(*ngram_range)
        ngram_deps = []

        for sentence in doc.sents:
            sentence_unigram_deps = [token.dep_ if token.dep_ != '' else 'Unk' for token in sentence]
            for n in list(range(ngram_range.start, ngram_range.stop+1)):
                sentence_ngram_deps = ['-'.join(ngram) for ngram in list(ngrams(sentence_unigram_deps, n))]
                ngram_deps.extend(sentence_ngram_deps)

        return ngram_deps


    def fit(self, documents, y=None):
        self.vectorizer.fit(documents)
        return self

    def transform(self, documents, y=None):
        self.count_deps(documents)
        dep_features = self.vectorizer.transform(documents)
        features_num =dep_features.shape[1]
        return dep_features

    def fit_transform(self, documents, y=None):
        self.count_deps(documents)
        dep_features = self.vectorizer.fit_transform(documents)

        return dep_features
    
    def count_deps(self, documents):
        if not hasattr(self, 'n_training_terms'):
            self.training_words = self.counter.fit_transform(documents)
            self.n_training_terms = self.training_words.sum(axis=1).getA().flatten()
        else:
            self.test_words = self.counter.transform(documents)
            self.n_test_terms = self.test_words.sum(axis=1).getA().flatten()

    
class FeatureSetReductor:
    def __init__(self, feature_extractor, measure=chi2, k=5000, k_ratio=1.0, normalize=True, oversample=True):
        self.feature_extractor = feature_extractor
        self.k = k
        self.k_ratio = k_ratio
        self.measure = measure
        self.normalize = normalize 
        self.oversample = oversample
        self.is_sparse = True
        if self.normalize:
            self.normalizer = Normalizer()
        
    def __str__(self) -> str:
        return( f'FeatureSetReductor for {self.feature_extractor}' )


    def fit(self, documents, y_dev=None):
        return self.feature_extractor.fit(documents, y_dev)

    def transform(self, documents, y_dev=None):
        matrix = self.feature_extractor.transform(documents)

        if self.normalize:
            matrix_norm  = self.normalizer.transform(matrix) 
            matrix_red = self.feat_sel.transform(matrix_norm)
        else:
            matrix_red = self.feat_sel.transform(matrix, y_dev)
        return matrix_red 

    def fit_transform(self, documents, y_dev=None):
        matrix = self.feature_extractor.fit_transform(documents, y_dev)
        self.features_in = matrix.shape[1]

        if self.features_in < self.k:
            self.k = self.features_in
        else:
            self.k = round(self.features_in * self.k_ratio) #keep k_ratio% of features

        self.feat_sel = SelectKBest(self.measure, k=self.k)

        if self.normalize:
            matrix_norm  = self.normalizer.fit_transform(matrix, y_dev)
            matrix_red = self.feat_sel.fit_transform(matrix_norm, y_dev)
            
        else:
            matrix_red = self.feat_sel.fit_transform(matrix, y_dev)

        return matrix_red
    
    def oversample_DRO(self, Xtr, ytr, Xte, yte, groups=None, rebalance_ratio=0.2, test_samples=100):
        if not isinstance(ytr, np.ndarray):
            ytr = np.array(ytr)
        self.dro = DistributionalRandomOversampling(rebalance_ratio=rebalance_ratio)
        samples = self.dro._samples_to_match_ratio(ytr)
        original_indices = self.dro.get_original_indices(Xtr, samples)
        y_oversampled = self.dro._oversampling_observed(ytr, samples)
        Xtr_old = Xtr.copy()

        if groups:
            groups = [group.split('_0')[0] for group in groups]
            groups_oversampled = []
            for group, i in zip(groups, samples):
                groups_oversampled.extend([group]*i)

        n_examples = samples.sum() - len(ytr)

        if hasattr(self.feature_extractor, 'n_training_terms'):
            print('Oversampling positive class using DRO method')
            self.n_training_terms =  self.feature_extractor.n_training_terms
            self.n_test_terms = self.feature_extractor.n_test_terms

            positives = ytr.sum()
            nD = len(ytr) 

            print('Before oversampling')
            print(f'positives = {positives} (prevalence={positives*100/nD:.2f}%)')

            Xtr, ytr = self.dro.fit_transform(Xtr, ytr, self.n_training_terms)
            Xte = self.dro.transform(Xte, self.n_test_terms, samples=test_samples) #new

            positives = ytr.sum()
            nD = len(ytr)
        
        else:

            Xtr = [Xtr[i] for i in original_indices]
            ytr = [ytr[i] for i in original_indices]

            Xtr = np.array(Xtr)
            Xte = np.array(Xte)
            
            if len(Xtr.shape) == 1:
                Xtr = Xtr.reshape(-1, 1)
            
             # Oversample Xte and yte to match test_samples
            Xte = np.tile(Xte, (test_samples, 1))  # Duplicate Xte to match test_samples
            yte = np.array([yte] * test_samples)  # Duplicate yte to match test_samples

        return Xtr, ytr, Xte, yte, groups_oversampled


class HstackFeatureSet:
    def __init__(self, feats=None, *vectorizers):
        self.vectorizers = vectorizers

    def fit(self, documents, authors=None):
        for v in self.vectorizers:
            v.fit(documents, authors)
        return self

    def transform(self, documents, authors=None):
        feats = [v.transform(documents, authors) for v in self.vectorizers]
        return self._hstack(feats)

    def fit_transform(self, documents, authors=None):
        feats = [v.fit_transform(documents, authors) for v in self.vectorizers]
        return self._hstack(feats)

    def _hstack(self, feats):
        for i, f in enumerate(feats):
            if not issparse(f):
                if not (isinstance(f, np.ndarray) and f.dtype == np.float64): 
                    feats[i] = np.asarray(f).astype(np.float64)

        anysparse = any(map(issparse, feats))
        if anysparse:
            feats = [csr_matrix(f) for f in feats]
            feats = hstack(feats)
        else:
            feats = np.hstack(feats)
        return feats