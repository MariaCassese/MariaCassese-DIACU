"""import json
import re

# 1. Pattern definiti
pattern_num_letter = r'\b\d{1,4}[a-zA-Zа-яА-Я]\b'  # es. 242v, 9a, 3р
pattern_only_numbers = r'\b\d{1,3}\b'              # solo numeri < 1000
# Pattern finale: riferimenti biblici come (Пс 116:1) o (ГИМ №1063)
pattern_biblical = r'\({1,2}[А-Я][а-я]{1,3}\s+\d{1,4}:\d{1,4}\){1,2}'
pattern_museum = r'\([А-ЯЁ]{2,6}\s+№\d{1,4}\)'

# 2. Carica il JSON
with open('ocs_data_with_epoch.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data.get("Documents", [])
                    
# 4. Rimuovi i riferimenti biblici e museali dal content, e salva nuovo JSON
for doc in docs:
    content = doc.get("Content", "")
    # Rimozione dei pattern
    content_cleaned = re.sub(pattern_biblical, '', content)
    content_cleaned = re.sub(pattern_num_letter, '', content)
    content_cleaned = re.sub(pattern_only_numbers, '', content)
    content_cleaned = re.sub(pattern_museum, '', content_cleaned)
    doc["Content"] = content_cleaned

# 5. Salva i documenti modificati in un nuovo file JSON
with open('ocs_data_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# 1. Carica il file JSON
with open('ocs_data_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data.get("Documents", [])

# 2. Pulisci i caratteri secondo le regole
for doc in docs:
    content = doc.get("Content", "")

    # Elimina sempre il bullet •
    content = content.replace("•", "")

    # Elimina il punto medio · solo se è tra due spazi
    content = re.sub(r'\s·\s', ' ', content)
    
    content = content.replace("(!)", "")

    # Aggiorna il campo
    doc["Content"] = content

# 3. Salva il file JSON modificato
with open('ocs_data_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# Regex per parentesi quadre e tonde
pattern_square = r'\[[^\[\]]+?\]'
pattern_round = r'\([^\(\)]+?\)'

# Lunghezza del contesto
context_window = 30

# Carica il JSON
with open('ocs_data_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data.get("Documents", [])

# Salva su file
with open('parentheses_occurrences.txt', 'w', encoding='utf-8') as out:
    for doc in docs:
        title = doc.get("Title", "Untitled")
        content = doc.get("Content", "")

        square_matches = list(re.finditer(pattern_square, content))
        round_matches = list(re.finditer(pattern_round, content))

        if square_matches or round_matches:
            out.write(f"\n--- {title} ---\n")

            if square_matches:
                out.write("[Parentesi quadre]\n")
                for match in square_matches:
                    start = max(match.start() - context_window, 0)
                    end = min(match.end() + context_window, len(content))
                    context = content[start:end].replace('\n', ' ')
                    out.write(f"{match.group()} --> ...{context}...\n")

            if round_matches:
                out.write("[Parentesi tonde]\n")
                for match in round_matches:
                    start = max(match.start() - context_window, 0)
                    end = min(match.end() + context_window, len(content))
                    context = content[start:end].replace('\n', ' ')
                    out.write(f"{match.group()} --> ...{context}...\n")"""

"""import json
import re

# Mappa dei pattern da cercare e rimuovere
patterns = {
    "(Мт :)": r"\(Мт :\)",
    "(Евр :)": r"\(Евр :\)",
    "(Пс., -)": r"\(Пс\., -\)",
    "(по Йо :)": r"\(по Йо :\)",
    "(по Мт , -)": r"\(по Мт , -\)",
}

# Carica il file JSON
with open("ocs_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("Documents", [])

# File di log delle occorrenze trovate
with open("parentheses_occurrences.txt", "w", encoding="utf-8") as out:
    for doc in docs:
        title = doc.get("Title", "Untitled")
        content = doc.get("Content", "")
        found_any = False

        for label, pattern in patterns.items():
            for match in re.finditer(pattern, content):
                found_any = True
                start = max(match.start() - 30, 0)
                end = min(match.end() + 30, len(content))
                context = content[start:end].replace("\n", " ")
                out.write(f"--- {title} ---\n")
                out.write(f"{label} trovato: {match.group()} --> ...{context}...\n")

        if found_any:
            out.write("\n")

# Rimozione dei pattern e salvataggio del file pulito
for doc in docs:
    content = doc.get("Content", "")
    for pattern in patterns.values():
        content = re.sub(pattern, '', content)
    doc["Content"] = content

# Salva il nuovo file JSON con i pattern rimossi
with open("ocs_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# Pattern: parentesi tonde o quadre con contenuto, con spazi ai lati
pattern_clean_parens = r'(?<=\s)[\[\(]([^\[\(\]\)]+)[\]\)](?=\s)'

# Carica JSON
with open("ocs_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("Documents", [])

# Logga le modifiche effettuate
with open("parentheses_occurrences.txt", "w", encoding="utf-8") as out:
    for doc in docs:
        title = doc.get("Title", "Untitled")
        content = doc.get("Content", "")
        matches = re.findall(pattern_clean_parens, content)
        if matches:
            out.write(f"\n--- {title} ---\n")
            for m in matches:
                out.write(f"{m}\n")

# Rimuove le parentesi mantenendo il contenuto
for doc in docs:
    content = doc.get("Content", "")
    content = re.sub(pattern_clean_parens, r'\1', content)
    doc["Content"] = content

# Salva il nuovo file pulito
with open("ocs_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""   

"""import json
import re

# Pattern: parentesi quadre con solo puntini
pattern_square_dots = r'\[\.+\]'

# Carica JSON
with open("ocs_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("Documents", [])

# Logga le occorrenze trovate con contesto
with open("parentheses_occurrences.txt", "w", encoding="utf-8") as out:
    for doc in docs:
        title = doc.get("Title", "Untitled")
        content = doc.get("Content", "")
        matches = list(re.finditer(pattern_square_dots, content))
        if matches:
            out.write(f"\n--- {title} ---\n")
            for match in matches:
                context = content[max(0, match.start() - 30):match.end() + 30].replace('\n', ' ')
                out.write(f"{match.group()} --> ...{context}...\n")

# Rimozione dei match
for doc in docs:
    content = doc.get("Content", "")
    content = re.sub(pattern_square_dots, '', content)
    doc["Content"] = content

# Salva nuovo file JSON
with open("ocs_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# Pattern: parentesi tonde dentro parole
pattern_parens_in_word = r'(\w*)\(([^\)]+)\)(\w*)'

# Carica JSON
with open("ocs_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("Documents", [])

# Logga i match trovati
with open("parentheses_occurrences.txt", "w", encoding="utf-8") as out:
    for doc in docs:
        title = doc.get("Title", "Untitled")
        content = doc.get("Content", "")
        matches = list(re.finditer(pattern_parens_in_word, content))
        if matches:
            out.write(f"\n--- {title} ---\n")
            for match in matches:
                full = match.group(0)
                context = content[max(match.start() - 30, 0):match.end() + 30].replace('\n', ' ')
                out.write(f"{full} --> ...{context}...\n")

# Applica la sostituzione per rimuovere le parentesi ma mantenere il contenuto
for doc in docs:
    content = doc.get("Content", "")
    content = re.sub(pattern_parens_in_word, r'\1\2\3', content)
    doc["Content"] = content

# Salva nuovo file
with open("ocs_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# Pattern da rimuovere solo se tra spazi
patterns = {
    "//": r'\s//\s',
    "/!/": r'\s/!/\s',
    "|": r'\s\|\s',
}

# Carica JSON
with open("ocs_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("Documents", [])

# Logga le occorrenze trovate
with open("parentheses_occurrences.txt", "w", encoding="utf-8") as out:
    for doc in docs:
        title = doc.get("Title", "Untitled")
        content = doc.get("Content", "")
        found = False
        for label, pattern in patterns.items():
            matches = list(re.finditer(pattern, content))
            for match in matches:
                found = True
                context = content[max(0, match.start()-30):match.end()+30].replace('\n', ' ')
                out.write(f"{label} in {title} --> ...{context}...\n")
        if found:
            out.write("\n")

# Rimuovi i segni se tra spazi
for doc in docs:
    content = doc.get("Content", "")
    for pattern in patterns.values():
        content = re.sub(pattern, ' ', content)  # sostituiamo con uno spazio singolo
    doc["Content"] = content

# Salva nuovo file
with open("ocs_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# 1. Carica il file JSON
with open('ocs_data_cleaned.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

docs = data.get("Documents", [])

# 2. Pulisci i caratteri secondo le regole
for doc in docs:
    content = doc.get("Content", "")

    # Elimina sempre il bullet •
    content = content.replace("•", "")

    # Elimina il punto medio · solo se è tra due spazi
    content = re.sub(r'\s·\s', ' ', content)

    # Aggiorna il campo
    doc["Content"] = content

# 3. Salva il file JSON modificato
with open('ocs_data_cleaned.json', 'w', encoding='utf-8') as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)"""
    
"""import json
import re

# 1‐letter pattern
pattern1 = re.compile(
    r'(?:(?<=\w)|(?<=\s)|(?<=^))'   # preceduto da lettera/spazio/inizio stringa
    r'\[([A-Za-zА-Яа-яЁё])\]'       # tra parentesi quadre 1 lettera
    r'(?:(?=\w)|(?=\s)|(?=$))'      # seguito da lettera/spazio/fine stringa
)

# 2‐letters pattern
pattern2 = re.compile(
    r'(?:(?<=\w)|(?<=\s)|(?<=^))'
    r'\[([A-Za-zА-Яа-яЁё]{2})\]'
    r'(?:(?=\w)|(?=\s)|(?=$))'
)

# 3‐letters pattern
pattern3 = re.compile(
    r'(?:(?<=\w)|(?<=\s)|(?<=^))'
    r'\[([A-Za-zА-Яа-яЁё]{3})\]'
    r'(?:(?=\w)|(?=\s)|(?=$))'
)

# Carica il JSON originale
with open("ocs_data_cleaned.json", "r", encoding="utf-8") as f:
    data = json.load(f)

docs = data.get("Documents", [])

# 1) Log delle occorrenze
with open("matches.txt", "w", encoding="utf-8") as log:
    for doc in docs:
        title   = doc.get("Title", "<no title>")
        content = doc.get("Content", "")
        for name, pat in (("1-letter", pattern1),
                          ("2-letters", pattern2),
                          ("3-letters", pattern3)):
            for m in pat.finditer(content):
                start, end = m.start(), m.end()
                # contesto di 30 caratteri
                ctxt = content[max(0, start-30): end+30].replace("\n", " ")
                log.write(f"[{title}] {name}: {m.group(0)} → …{ctxt}…\n")

# 2) Rimuovi SOLO le parentesi [ ] per 1–3 lettere
for doc in docs:
    text = doc.get("Content", "")
    text = pattern3.sub(lambda m: m.group(1), text)
    text = pattern2.sub(lambda m: m.group(1), text)
    text = pattern1.sub(lambda m: m.group(1), text)
    doc["Content"] = text

# 3) Salva il JSON modificato
with open("ocs_data_cleaned.json", "w", encoding="utf-8") as f:
    json.dump({"Documents": docs}, f, ensure_ascii=False, indent=2)

print("Fatto! • matches.txt • ocs_data_cleaned.json creati.")"""






