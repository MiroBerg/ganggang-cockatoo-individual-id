import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# Load csv and model
test_data = pd.read_csv("...path_to.../testing_dataset.csv")
agesex_model = load_model("...path_to.../agesex_model.keras")

# Create columns
test_data["pred_age"] = ""
test_data["pred_sex"] = ""

# Age sex dictionary
as_dict = {"0": "a_f", "1": "a_m", "2": "j_f", "3": "j_m"}

# Predict agesex class on all testing images
for index, row in tqdm(test_data.iterrows(), total=len(test_data)):
    if float(row["seg_conf"]) > 0.8: # Only use high confidence predictions
        img_path = row["file_path"] 
    
        # Load and preprocess image
        img = image.load_img(img_path, target_size=(384, 384))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    
        # Predict
        preds = agesex_model.predict(x, verbose=0)
        top_class = np.argmax(preds, axis=1)[0]
    
        top_class_label = as_dict[str(top_class)]
        test_data.at[index, "pred_age"] = top_class_label[0]
        test_data.at[index, "pred_sex"] = top_class_label[2]

# Create combined true and predicted labels
test_data["true_combined"] = test_data["age"] + "_" + test_data["sex"]
test_data["pred_combined"] = test_data["pred_age"] + "_" + test_data["pred_sex"]

# Age-sex classes
classes = ["a_f", "a_m", "j_f", "j_m"]

# Compute confusion matrix
cm = confusion_matrix(label_csv["true_combined"], label_csv["pred_combined"], labels=classes)

# Normalize by row (proportional)
cm_normalized = np.round((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]), 3)

# Show confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Adult\nFemale", "Adult\nMale", "Juvenile\nFemale", "Juvenile\nMale"])
disp.plot(cmap='Blues', colorbar=True, text_kw={"fontsize": 11})
disp.ax_.tick_params(axis='both', labelsize=11)
disp.im_.colorbar.ax.set_ylabel("Proportion", rotation=90, fontsize=10)
plt.xlabel("Predicted class", fontsize=10)
plt.ylabel("True class", fontsize=10)
plt.tight_layout()
plt.show()

# Show classification report
report = classification_report(label_csv["true_combined"], label_csv["pred_combined"], labels=classes, digits=3)
print(report)
