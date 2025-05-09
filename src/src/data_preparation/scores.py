import pandas as pd
import ast
from sklearn.metrics import classification_report

# 1. Leggi il CSV
df = pd.read_csv("results/results_oversampling_due classi.csv")
# Aggiusta la formattazione delle confusion matrix
def fix_cm_string(s):
    # Rimuove eventuali spazi iniziali/finali e aggiunge virgole tra numeri
    s = s.strip().replace('[', '').replace(']', '')
    items = s.split()
    return [int(x) for x in items]

# Applica la correzione
df["Confusion matrix"] = df["Confusion matrix"].apply(fix_cm_string)

# 3. Ricostruisci y_true e y_pred
y_true = []
y_pred = []

for cm in df["Confusion matrix"]:
    tn, fp, fn, tp = cm
    if tp == 100:
        y_true.append(1)
        y_pred.append(1)
    elif tn == 100:
        y_true.append(0)
        y_pred.append(0)
    elif fp == 100:
        y_true.append(0)
        y_pred.append(1)
    elif fn == 100:
        y_true.append(1)
        y_pred.append(0)

# 4. Calcola il classification report
report = classification_report(y_true, y_pred, digits=2)
print(report)