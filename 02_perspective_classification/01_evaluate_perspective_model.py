import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Load csv and model
val_data = pd.read_csv("...path_to.../validation_dataset.csv")
agesex_model = load_model("...path_to.../perspective_model.keras")

# Create columns
test_data["pred_persp"] = ""

# Perspective dictionary
persp_dict = {"0": "f", "1": "r", "2": "l", "3": "b"}

# Predict perspective on all validation images
for index, row in tqdm(val_data.iterrows(), total=len(val_data)):
    if float(row["seg_conf"]) > 0.8: # Only use high confidence segmentations
        img_path = row["file_path"] 
    
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    
        # Predict
        preds = agesex_model.predict(x, verbose=0)
        top_class = np.argmax(preds, axis=1)[0]
    
        top_class_label = persp_dict[str(top_class)]
        val_data.at[index, "pred_persp"] = top_class_label[0]

# Age-sex classes
classes = ["f", "r", "l", "b"]

# Compute confusion matrix
cm = confusion_matrix(label_csv["true_persp"], label_csv["pred_persp"], labels=classes)

# Normalize by row (proportional)
cm_normalized = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]), 3)

# Show confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Front", "Left", "Right", "Back"])
disp.plot(cmap='Blues', colorbar=True, text_kw={"fontsize": 11})
disp.ax_.tick_params(axis='both', labelsize=11)
disp.im_.colorbar.ax.set_ylabel("Proportion", rotation=90, fontsize=10)
plt.xlabel("Predicted class", fontsize=10)
plt.ylabel("True class", fontsize=10)
plt.tight_layout()
plt.show()

# Show classification report
report = classification_report(val_data["true_persp"], val_data["pred_persp"], labels=classes, digits=3)
print(report)
