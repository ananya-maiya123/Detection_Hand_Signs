from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_confusion_matrix(y_true, y_pred, classes):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure
    plt.figure(figsize=(15, 15))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('Model/confusion_matrix.png')
    plt.close()

def plot_accuracy_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('Model/accuracy_history.png')
    plt.close()

# Load the trained model
model = load_model("Model/best_model.h5")  # Using the best model saved during training

# Compile the model with appropriate configurations
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Path to test dataset
test_data_dir = "data"
img_size = 224  # Match the size used in training
batch_size = 32

# Prepare the test data generator
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Get class indices and create a mapping
class_indices = test_generator.class_indices
class_names = list(class_indices.keys())

# Evaluate the model
print("Evaluating model...")
loss, accuracy = model.evaluate(test_generator)
print(f"\nOverall Test Results:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

# Generate and print classification report
print("\nDetailed Classification Report:")
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print(report)

# Save classification report to CSV
report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.to_csv('Model/classification_report.csv')

# Plot confusion matrix
print("\nGenerating confusion matrix...")
plot_confusion_matrix(y_true, y_pred, class_names)

# Print per-class accuracy
print("\nPer-class Accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    class_accuracy = np.mean(y_pred[class_mask] == i)
    print(f"{class_name}: {class_accuracy * 100:.2f}%")

print("\nEvaluation complete! Results have been saved to the Model directory.")
print("- Confusion matrix: Model/confusion_matrix.png")
print("- Detailed report: Model/classification_report.csv")
