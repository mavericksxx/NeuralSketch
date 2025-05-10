import numpy as np
import re
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

categories = [
    'apple', 'book', 'bowtie', 'candle', 'cloud', 'cup', 'door', 
    'envelope', 'eyeglasses', 'guitar', 'hammer', 'hat', 'ice cream', 
    'leaf', 'scissors', 'star', 't-shirt', 'pants', 'lightning', 'tree'
]

with open('logs.txt', 'r') as f:
    logs = f.read()

matrices = re.findall(r'Test confusion matrix:([\s\S]*?)(?=Epoch:|$)', logs)

matrix_lines = matrices[0].strip().split('\n')
joined_lines = [matrix_lines[i] + matrix_lines[i+1] for i in range(0, len(matrix_lines), 2)]

cm = []
for line in joined_lines:
    nums = [int(n) for n in line.replace('[', '').replace(']', '').split()]
    cm.append(nums)
cm_array = np.array(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm_array, 
                             display_labels=categories)
fig, ax = plt.subplots(figsize=(15,15))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=90, values_format='d')
plt.title("20-Category Sketch Recognition Confusion Matrix")
plt.savefig('confusion_matrix.png', bbox_inches='tight')