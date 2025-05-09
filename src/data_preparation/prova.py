import json
import re

with open('ocs_data_with_epoch.json', 'r', encoding='utf-8') as f:
    full = json.load(f)

docs = full.get('Documents', [])
if not docs:
    raise ValueError("The field 'Documents' doesn't exist")

target_epochs = {'Old Church Slavonic', 'Church Slavonic'}

filtered_docs = [
    doc for doc in docs
    if doc.get('Epoch') in target_epochs and not re.search(r'\d', doc.get('Title', ''))
]

output = {'Documents': filtered_docs}

with open('ocs_cs_all_filtered.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)




    
    
    