import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5, 6]
train_loss = [0.3527, 0.1715, 0.1399, 0.1196, 0.1089, 0.0828]
train_accuracy = [0.9026, 0.9467, 0.9554, 0.9612, 0.966, 0.972]
val_loss = [0.1759, 0.1576, 0.1534, 0.1571, 0.1556, 0.1758]
val_accuracy = [0.9475, 0.9539, 0.9543, 0.9532, 0.9563, 0.9523]

plt.figure(figsize=(10, 6))
plt.style.use('seaborn-v0_8-whitegrid')

plt.plot(epochs, train_loss, marker='o', color='#35b0ab', linewidth=2, label='Training Loss')
plt.plot(epochs, val_loss, marker='o', color='#f8a978', linewidth=2, label='Validation Loss')
plt.plot(epochs, train_accuracy, marker='o', color='#a52a2a', linewidth=2, label='Training Accuracy')
plt.plot(epochs, val_accuracy, marker='o', color='#f0e68c', linewidth=2, label='Validation Accuracy')

plt.title('Training and Validation Metrics Over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

plt.axvline(x=3, color='gray', linestyle='--', alpha=0.5)
plt.text(3.1, 0.4, 'Overfitting begins', fontsize=10)

plt.savefig('training_curves.png', dpi=300)
plt.show()

