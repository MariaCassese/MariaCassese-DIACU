import json
from pathlib import Path
from tqdm import tqdm
import re
import ipdb

def get_slavonic_function_words():
    return []
    

def load_corpus_json(json_path: str, **filters) -> tuple[list[str], list[str], list[str]]:
    """
    Carica il JSON e, se skip_ruthenians=True, salta tutti i documenti
    il cui campo 'Epoch' è 'Ruthenian'.
    Ritorna liste parallele di (testi, epoche, titoli).
    """
    # 1) Apri e leggi il JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raw_docs = data.get('Documents', [])
    
    corpus = []
    # 2) Cicla sui documenti
    for doc in tqdm(raw_docs, desc=f'Loading from {Path(json_path).name}'):
        if _should_skip_file(doc['Title'], filters):
            
            print(f'Removing {doc.name}')
            continue
        
        # 2b) Prendi il contenuto e salta se è vuoto
        content = doc.get('Content', '').strip()
        if not content:
            continue
        
        # 2c) Estrai metadati
        title    = doc.get('Title',    '').strip()
        epoch    = doc.get('Epoch',    '').strip()
        language = doc.get('Language', '').strip()
        area     = doc.get('Area',     '').strip()
        
        # 2d) Costruisci un filename “sicuro”
        filename = title.replace(' ', '_')
        
        # 2e) Pulisci il testo
        text = _clean_text(content)
        
        corpus.append({
            'text':     text,
            'title':    title,
            'epoch':    epoch,
            'language': language,
            'area':     area,
            'filename': filename
        })
    
    # 3) Estrai le liste finali
    documents = [d['text']  for d in corpus]
    epochs    = [d['epoch'] for d in corpus]
    filenames    = [d['title'] for d in corpus]
    
    print(f'Total documents: {len(documents)}')
    return documents, epochs, filenames

def _should_skip_file(filename: str, filters: dict) -> bool:
    """Check if file should be filtered out based on criteria."""
    checks = {
        'remove_test': lambda f: "Oustav' stgo i văselenskago iže v Konstantini gradě šestago săbora" in f.lower(),
    }
    return any(check(filename) for flag, check in checks.items() if filters.get(flag))

def _should_skip_file(filename: str, filters: dict) -> bool:
    """Check if file should be filtered out based on criteria."""
    checks = {
        'remove_test': lambda f: filters.get('remove_test', False) 
                                  and "Oustav' stgo i văselenskago iže v Konstantini gradě šestago săbora" in f.lower(),
        'test_document': lambda f: filters.get('test_document') 
                                   and f.strip() == filters['test_document'].strip()
    }
    return any(check(filename) for flag, check in checks.items() if filters.get(flag))

def _clean_text(text: str) -> str:
    text = text.lower()
    #text = re.sub(r'\{[^{}]*\}', '', text)
    #text = re.sub(r'\*[^**]*\*', '', text) 
    text = re.sub(r'<\w>(.*?)</\w>', r'\1', text)
    text = text.replace('\x00', '')
    return text.strip()


"""docs, epochs, titles = load_corpus_json(
    'ocs_data_with_epoch.json',
    skip_ruthenians = False
)"""

        
        
        
    

