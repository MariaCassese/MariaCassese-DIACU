import concurrent.futures
import re
import argparse
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import csv
import time
from tqdm import tqdm
import spacy
import nltk
from nltk import sent_tokenize


from data_preparation.data_loader import load_corpus_json
from data_preparation.segmentation import Segmentation
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, 
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    make_scorer
)

from feature_extraction.features import (
    DocumentProcessor,
    FeaturesMendenhall,
    FeaturesSentenceLength,
    FeaturesPOST,
    FeatureSetReductor,
    HstackFeatureSet,
    FeaturesCharNGram,
)


@dataclass
class ModelConfig:
    """Configuration for the model training and evaluation"""
    processes: int = 20
    n_jobs: int = 10
    segment_min_token_size: int = 400
    random_state: int = 0
    k_ratio: float = 0.8
    oversample: bool = False
    rebalance_ratio: float = 0.5
    save_res: bool = True
    results_filename: str = 'results_base_binary_cleaned_4classes_new_4.csv'
    results_path: str = './results/4classes_new_4'   

    @classmethod
    def from_args(cls):
        """Create config from command line args"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--results-filename', default='results_base_binary_cleaned_4classes_new_4.csv',
                    help='Filename for saving results')
        parser.add_argument('--results-path', 
                    default='./results/4classes_new_4/',
                    help='Directory path for saving results')
        args = parser.parse_args()
            
        config = cls()
        config.results_filename = args.results_filename
        config.results_path = args.results_path
        
        return config
            
class EpochshipVerification:
    """Main class for epochship verification system"""
    
    def __init__(self, config: ModelConfig, nlp: spacy.Language):
        self.config = config
        self.nlp = nlp
        self.accuracy = 0
        self.posterior_proba = 0
        
    def load_dataset(self,  
                     path: str = './data/DIACU_1.0.json'
                     ) -> Tuple[List[str], List[str], List[str]]:
        
        print('Loading data...')
        documents, epochs, filenames = load_corpus_json(
            json_path=path, 
            skip_ruthenians=False,
        )
        print('Data loaded.')
        print('Labels:',Counter(epochs))
        return documents, epochs, filenames
    
    def loo_split(self, i: int, X: List[str], y: List[int], doc: str, ylabel: int, 
        filenames: List[str]) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str]]:
        doc_name = filenames[i]
        print(f'Test document: {doc_name[:-2]}')
        
        X_test = [doc]
        X_dev = list(np.delete(X, i))
        y_test = [int(ylabel)]
        y_dev = list(np.delete(y, i))
        # y_test = [1]
        # y_dev = [1 if label == ylabel else 0 for label in np.delete(y, i)]
        groups_dev = list(np.delete(filenames, i))
        
        return X_dev, X_test, y_dev, y_test, groups_dev, [doc_name]
    
    def segment_data(self, X_dev: List[str], X_test: List[str], y_dev: List[int], 
                    y_test: List[int], groups_dev: List[str], groups_test: List[str]
                    ) -> Tuple[List[str], List[str], List[int], List[int], List[str], List[str], List[str]]:
        """Segment the documents into smaller chunks"""
        
        print('Segmenting data...')
        whole_docs_len = len(y_test)

        segmentator_dev = Segmentation(
            split_policy='by_sentence',
            tokens_per_fragment=self.config.segment_min_token_size,
        )
        splitted_docs_dev = segmentator_dev.fit_transform(
            documents=X_dev,
            epochs=y_dev,
            filenames=groups_dev
        )

        segmentator_test = Segmentation(
            split_policy='by_sentence',
            tokens_per_fragment=self.config.segment_min_token_size, 
        )
        splitted_docs_test = segmentator_test.transform(
            documents=X_test,
            epochs=y_test,
            filenames=groups_test
        )
        groups_test = segmentator_test.groups

        X_dev = splitted_docs_dev[0]
        y_dev = splitted_docs_dev[1]
        groups_dev = segmentator_dev.groups

        X_test = splitted_docs_test[0][:whole_docs_len]
        y_test = splitted_docs_test[1][:whole_docs_len]
        groups_test_entire_docs = groups_test[:whole_docs_len]

        X_test_frag = splitted_docs_test[0][whole_docs_len:]
        y_test_frag = splitted_docs_test[1][whole_docs_len:]
        groups_test_frag = groups_test[whole_docs_len:]

        print('Segmentation complete.')
        
        return (X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, 
                groups_dev, groups_test_entire_docs, groups_test_frag)
        
    def get_processed_documents(self, documents: List[str], filenames: List[str], 
                              processed: bool = False, 
                              cache_file: str = '.cache/processed_docs.pkl') -> Dict[str, spacy.tokens.Doc]:
        """Process documents using spaCy"""
        print('Processing documents...')
        
        if not processed:
            self.nlp.max_length = max(len(doc) for doc in documents)
            processor = DocumentProcessor(language_model=self.nlp, savecache=cache_file)
            processed_docs = processor.process_documents(documents, filenames, self.config.processes)
        else:
            processor = DocumentProcessor(savecache=cache_file)
            processed_docs = processor.process_documents(documents, filenames, self.config.processes)
        
        return processed_docs
    
    
    def find_segment(self, segment: str, processed_document: spacy.tokens.Doc) -> spacy.tokens.Span:
        """Find a segment within a processed document"""

        start_segment = sent_tokenize(segment)[0]
        start_idx = processed_document.text.find(start_segment)
        end_idx = start_idx + len(segment)
        
        processed_seg = processed_document.char_span(start_idx, end_idx, alignment_mode='expand')
        if not processed_seg:
            processed_seg = processed_document.char_span(start_idx, end_idx-1, alignment_mode='expand')
        
        return processed_seg

    def get_processed_segments(self, processed_docs: Dict[str, spacy.tokens.Doc], 
                             X: List[str], groups: List[str], dataset: str = ''
                             ) -> List[Union[spacy.tokens.Doc, spacy.tokens.Span]]:
        """Extract processed segments from documents"""
        print(f'Extracting processed {dataset} segments...')
        
        none_count = 0
        processed_X = []
        
        for segment, group in tqdm(zip(X, groups), total=len(X), desc='Progress'):
            if re.search(r'_\d+_\d+$', group):  # entire doc
                processed_doc = processed_docs[re.sub(r'_\d+_\d+$', '', group)]
                processed_X.append(processed_doc)
            else:  # segment
                group_idx = group.find('_0')
                group_key = group[:group_idx]
                ent_doc_processed = processed_docs[group_key]
                processed_segment = self.find_segment(segment, ent_doc_processed)
                
                if not processed_segment:
                    none_count += 1
  
                processed_X.append(processed_segment)
        
        print(f'None count: {none_count}\n')
        return processed_X

    def extract_feature_vectors(self, processed_docs_dev: List[spacy.tokens.Doc], 
                              processed_docs_test: List[spacy.tokens.Doc],
                              y_dev: List[int], y_test: List[int], 
                              groups_dev: List[str]) -> Tuple[np.ndarray, ...]:
        
        print('Extracting feature vectors...')

        vectorizers = [
            FeaturesPOST(n=(1,3)),
            FeaturesMendenhall(upto=20),
            FeaturesSentenceLength(),
            FeaturesCharNGram(n=(1,3))
        ]

        hstacker = HstackFeatureSet(vectorizers)
        feature_sets_dev = []
        feature_sets_test = []
        feature_sets_dev_orig = []
        feature_sets_test_orig = []
        orig_groups_dev = groups_dev.copy()

        for vectorizer in vectorizers:
            print(f'\nExtracting {vectorizer}')
            reductor = FeatureSetReductor(
                vectorizer, 
                k_ratio=self.config.k_ratio
            )

            print('\nProcessing development set')
            features_set_dev = reductor.fit_transform(processed_docs_dev, y_dev)
            
            print('\nProcessing test set')
            features_set_test = reductor.transform(processed_docs_test)

            if self.config.oversample:
                feature_sets_dev_orig.append(features_set_dev)
                feature_sets_test_orig.append(features_set_test)
                orig_y_dev = y_dev.copy()

                (features_set_dev, y_dev_oversampled, features_set_test, 
                 y_test_oversampled, groups_dev) = reductor.oversample_DRO(
                    Xtr=features_set_dev,
                    ytr=y_dev,
                    Xte=features_set_test,
                    yte=y_test,
                    groups=orig_groups_dev,
                    rebalance_ratio=self.config.rebalance_ratio
                )
                feature_sets_dev.append(features_set_dev)
                feature_sets_test.append(features_set_test)
            else:
                feature_sets_dev.append(features_set_dev)
                feature_sets_test.append(features_set_test)

        orig_feature_sets_idxs = self._compute_feature_set_idx(
            vectorizers, 
            feature_sets_dev_orig
        )
        feature_sets_idxs = self._compute_feature_set_idx(
            vectorizers, 
            feature_sets_dev
        )

        print(f'Feature sets computed: {len(feature_sets_dev)}')
        print('\nStacking feature vectors')

        if len(feature_sets_dev_orig)>0:
            X_dev_stacked_orig = hstacker._hstack(feature_sets_dev_orig)
            X_test_stacked_orig = hstacker._hstack(feature_sets_test_orig)

        X_dev_stacked = hstacker._hstack(feature_sets_dev)
        X_test_stacked = hstacker._hstack(feature_sets_test)


        y_dev_final = y_dev_oversampled if self.config.oversample else y_dev
        y_test_final = y_test_oversampled if self.config.oversample else y_test

        if self.config.oversample:
            return (X_dev_stacked, X_test_stacked, y_dev_final, y_test_final, 
                   groups_dev, feature_sets_idxs, orig_feature_sets_idxs,
                   X_dev_stacked_orig, X_test_stacked_orig, orig_y_dev, 
                   orig_groups_dev)
        else:
            return (X_dev_stacked, X_test_stacked, y_dev_final, y_test_final,
                   groups_dev, feature_sets_idxs, None, None, None, None, None)
                   
    def _compute_feature_set_idx(self, vectorizers, feature_sets_dev):
        """Helper method to compute feature set indices"""
        start_idx = 0
        end_idx = 0
        feature_sets_idxs = {}
        
        for vect, fset in zip(vectorizers, feature_sets_dev):
            if isinstance(fset, list):
                fset = np.array(fset)
            
            if len(fset.shape) == 1:
                fset = fset.reshape(-1, 1)
            
            feature_shape = fset.shape[1]
            end_idx += feature_shape
            feature_sets_idxs[vect] = (start_idx, end_idx)
            start_idx = end_idx
            
        return feature_sets_idxs
    
    def train_model(self, X_dev: np.ndarray, y_dev: List[int], 
                    groups_dev: List[str], model: BaseEstimator, 
                    model_name: str) -> BaseEstimator:
            
        param_grid = {
            'C': np.logspace(-4, 4, 9),
            'class_weight': ['balanced', None],
        }
        
        cv = StratifiedGroupKFold(
            n_splits=5,
            shuffle=True,
            random_state=self.config.random_state
        )
        f1 = make_scorer(f1_score, average='micro', zero_division=0)
        
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            n_jobs=self.config.n_jobs,
            scoring=f1,
            verbose=True
        )
        
        grid_search.fit(X_dev, y_dev, groups=groups_dev)
        print(f'Model fitted. Best params: {grid_search.best_params_}')
        print(f'Best scores: {grid_search.best_score_}\n')
        
        return grid_search.best_estimator_

    def evaluate_model(self, clf: BaseEstimator, X_test: np.ndarray, 
                    y_test: List[int], return_proba: bool = True
                    ) -> Tuple[float, float, np.ndarray, float]:
        
        print('Evaluating performance...',
            '(on fragmented text)' if len(y_test) > 110 else '\n')
        
        y_test = np.array(y_test * X_test.shape[0])
        y_pred = clf.predict(X_test)
        
        if return_proba:
            probabilities = clf.predict_proba(X_test)
            self.posterior_proba = np.median(
                [prob[class_idx] for prob, class_idx in zip(probabilities, y_pred)]
            )
            print(f'Posterior probability: {self.posterior_proba}')
        
        self.accuracy = accuracy_score(y_test, y_pred)
        try:
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=1.0)
        except:
            f1 = 0.0
            print("F1 score could not be computed, setting to 0.0")
        try:
            precision, recall, _, _ = precision_recall_fscore_support(
                y_test, y_pred, average='binary', zero_division=1.0
            )
        except:
            precision, recall = 0.0, 0.0
            print("Precision and recall could not be computed, setting to 0.0")
        print(">>> shape X_test:", X_test.shape)
        print(">>> len y_test:", len(y_test))
        assert len(y_test) == X_test.shape[0], "y_test and X_test must have the same length!"
        try:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
        except:
            cm = np.array([[0, 0], [0, 0]])
        # cf = np.array([tn, fp, fn, tp])

        
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'Accuracy: {self.accuracy}')
        print(f'F1: {f1}\n')
        # print(classification_report(y_test, y_pred, zero_division=1.0))
        # print(f'\nConfusion matrix: (tn, fp, fn, tp)\n{cf}\n')
        """print(f"Random seed: {self.config.random_state}")"""
        
        return self.accuracy, f1, cm, self.posterior_proba, y_pred

    def save_results(self, ylabel, accuracy: float, f1: float,
                    posterior_proba: float, y_pred, cm, model_name: str, 
                    doc_name: str, features: List[str], 
                    file_name: str, path_name="output.json"):
        
        path = Path(path_name)
        print(f'Saving results in {file_name}\n')
        
        data = {
            'Document test': doc_name,
            'Label': ylabel,
            'Prediction': y_pred,
            'Accuracy': accuracy,
            'Proba': posterior_proba,
            'Confusion matrix': cm
        }
        
        output_path = path / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
        
        print(f"{model_name} results saved in {file_name}\n")


    def run(self, save_results: bool = True, 
            filter_dataset: bool = False):
        """Run the complete epochship verification process"""
        start_time = time.time()
        print(f'Start time: {time.strftime("%H:%M")}')
        print(f'Building LOO model\n')

        documents, epochs, filenames = self.load_dataset()
        
        filenames = [f'{filename}_0' for filename in filenames]

        processed_documents = self.get_processed_documents(documents, filenames)
        
        mapping = {
            'Ruthenian': 3,
            'New Church Slavonic':  2,
            'Old Church Slavonic':  1,
            'Church Slavonic':  0,
        }
        
        y = [
            mapping[e.strip()]
            for e in epochs
        ]

        print("Class balance:", np.unique(y, return_counts=True))
        
        test_indices = list(range(len(filenames)))

        # for i in test_indices:
        #     self._process_single_document(
        #         i, documents, y, processed_documents, filenames,
        #         save_results, self.config.results_filename,
        #         self.config.results_path
        #         )
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.processes) as executor:
            futures = [
                executor.submit(
                    self._process_single_document,
                    i, documents, y, processed_documents, filenames,
                    save_results, self.config.results_filename,
                    self.config.results_path
                )
                for i in test_indices
            ]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        total_time = round((time.time() - start_time) / 60, 2)
        print(f'Total time spent for model building: {total_time} minutes.')
    
    def _process_single_document(self, i: int, documents: List[str], y: List[int], 
                              processed_documents: Dict[str, spacy.tokens.Doc],
                              filenames: List[str], save_results: bool,
                              file_name: str, path_name: str):
                                  
        """Process a single document for epochship verification"""
        start_time_single_iteration = time.time()
        np.random.seed(self.config.random_state)
        
        doc, ylabel = documents[i], y[i]
        X_dev, X_test, y_dev, y_test, groups_dev, groups_test = self.loo_split(
            i, documents, y, doc, ylabel, filenames
        )

        (X_dev, X_test, y_dev, y_test, X_test_frag, y_test_frag, groups_dev, 
         groups_test, groups_test_frag) = self.segment_data(
            X_dev, X_test, y_dev, y_test, groups_dev, groups_test
        )
        print(np.unique(y, return_counts=True))
        
        X_dev_processed = self.get_processed_segments(
        processed_documents, X_dev, groups_dev, dataset='training'
        )
        # filter None in training
        filtered = [
            (seg, lbl, grp)
            for seg, lbl, grp in zip(X_dev_processed, y_dev, groups_dev)
            if seg is not None
        ]
        if not filtered:
            raise ValueError("Tutti i frammenti di training sono None!")
        X_dev_processed, y_dev, groups_dev = map(list, zip(*filtered))

        X_test_processed = self.get_processed_segments(
            processed_documents, X_test, groups_test, dataset='test'
        )
        # filter None in test
        filtered_test = [
            (seg, lbl)
            for seg, lbl in zip(X_test_processed, y_test)
            if seg is not None
        ]
        if not filtered_test:
            raise ValueError("Tutti i segmenti di test sono None!")
        X_test_processed, y_test = map(list, zip(*filtered_test))

        X_test_frag_processed = self.get_processed_segments(
            processed_documents, X_test_frag, groups_test_frag, 
            dataset='test fragments'
        )
        # filter None in test fragments
        filtered_frag = [
            (seg, lbl, grp)
            for seg, lbl, grp in zip(X_test_frag_processed, y_test_frag, groups_test_frag)
            if seg is not None
        ]
        if filtered_frag:
            X_test_frag_processed, y_test_frag, groups_test_frag = map(list, zip(*filtered_frag))
        else:
            X_test_frag_processed, y_test_frag, groups_test_frag = [], [], []


        X_len = len(X_dev_processed)
        print(f'X_len: {X_len}')

        (X_dev, X_test, y_dev, y_test, groups_dev, feature_idxs, 
         original_feature_idxs, original_X_dev, original_X_test, 
         orig_y_dev, orig_groups_dev) = self.extract_feature_vectors(
            X_dev_processed, X_test_processed, y_dev, y_test, 
            groups_dev
        )

        models = [
            (LogisticRegression(
                random_state=self.config.random_state, 
                solver='sag',
            ), 'Logistic Regressor')
        ]

        for model, model_name in models:
            print(f'\nBuilding {model_name} classifier...\n')
            clf = self.train_model(X_dev, y_dev, groups_dev, model, model_name)
            acc, f1, cm, posterior_proba, y_pred = self.evaluate_model(
                clf, X_test, y_test
            )

            if save_results:
                self.save_results(ylabel, acc, f1, posterior_proba, y_pred, cm, model_name,
                    groups_test[0][:-2], feature_idxs.keys(),
                    file_name+f"_{i}", path_name)

        iteration_time = round((time.time() - start_time_single_iteration) / 60, 2)
        print(f'Time spent for model building for document {groups_test[0][:-2]}: {iteration_time} minutes.')

    

def main():
    config = ModelConfig.from_args()
    nlp = spacy.load('ru_core_news_lg')
    nltk.download('punkt_tab')
    av_system = EpochshipVerification(config, nlp)
    av_system.run(
        save_results=config.save_res,
        filter_dataset=False,
    )

if __name__ == '__main__':
    main()
