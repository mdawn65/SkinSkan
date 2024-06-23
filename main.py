import warnings
warnings.filterwarnings('ignore')

from openai import OpenAI
from PIL import Image
import matplotlib.pyplot as plt
import joblib

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import numpy as np
from PIL import Image
import os

import json

# Load the image
image_path = 'image.jpg'
img = Image.open(image_path)

# Display the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

API_KEY = os.getenv('KEY')

client = OpenAI(
    api_key=API_KEY
)

symptoms = """
"ease-of-tanning-sunburn-1-10": "BLANK",
"age": "BLANK",
"birth-assigned-sex": "BLANK",
"zipcode": "BLANK",
"itchy": "BLANK",
"burning": "BLANK",
"swelling": "BLANK",
"rashes": "BLANK",
"bumps": "BLANK",
"dry-skin": "BLANK",
"scaly-skin": "BLANK",
"peeling-skin": "BLANK"
"""

first = "Hi, I am NAME. Ask me questions about my skin."

messages = [
  {"role": "system", "content": """
  You are an assistant dermatologist trying to diagnose different skin
  conditions. Please ask the patient questions based on the JSON to help
    diagnose them and fill out. Only put true/false, male/female, or a number as your answers for the
    BLANKS, even if the user gives a location. Also only fill in one of the
    blanks each try. Don't put anything in the BLANK if the user has not responded yet.
    Only put in what the user responses with. Leave the BLANK a BLANK if user has not responded yet.
    Make sure to ask the questions in order.
    Here is the JSON template:

  Here is what the user will send:
    {{
      {{
        {0}
      }},
      "user-response": "{1}"
    }}

    Here is a sample response:
    {{
      "llm-question": "please ask a question to fill in one of the BLANKs",
      "data": {{
        {0}
      }}
    }}
  """.format(symptoms, first)}
]


for i in range(5):
  response = client.chat.completions.create(
    model="gpt-4o",
    response_format={ "type": "json_object" },
    temperature=0,
    seed=42,
    messages=messages
  )
  r = response.choices[0].message.content
  y = json.loads(r)
  print(y["llm-question"])
  messages.append({"role": "system", "content": r})
  u = input("USER:")
  messages.append({"role": "user", "content": """
  {{
    {{
      {0}
    }},
    "user-response": "{1}"
  }}
  """.format(r, u)})
  if "BLANK" not in r:
    break

y = json.loads(r)

if type(y["data"]["ease-of-tanning-sunburn-1-10"]) == str:
  y["data"]["ease-of-tanning-sunburn-1-10"] = int(y["data"]["ease-of-tanning-sunburn-1-10"])
y["data"]["ease-of-tanning-sunburn-1-10"] = "FST" + str(7 - max(1, min(6, int(6 * y["data"]["ease-of-tanning-sunburn-1-10"] / 10))))

if type(y["data"]["age"]) == str:
  y["data"]["age"] = int(y["data"]["age"])
age = y["data"]["age"]
if (age < 18):
  age_str = "AGE_UNKNOWN"
elif (age < 30):
  age_str = "AGE_18_TO_29"
elif (age < 40):
  age_str = "AGE_30_TO_39"
elif (age < 50):
  age_str = "AGE_40_TO_49"
elif (age < 60):
  age_str = "AGE_50_TO_59"
elif (age < 70):
  age_str = "AGE_60_Т0_69"
elif (age < 80):
  age_str = "AGE_70_TO_79"
else:
    age_str = "AGE_UNKNOWN"

y["data"]["age"] = age_str

y["data"]["birth-assigned-sex"] = y["data"]["birth-assigned-sex"].upper()

# Load the trained model from the .h5 file
model = tf.keras.models.load_model('hackathon_model_1.h5')  # Replace with your model file path

# Load the fitted column transformer
column_transformer = joblib.load('column_transformer.pkl')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_classes.npy', allow_pickle=True)

# Function to preprocess the example image from a local file
def preprocess_example_image(img_path):
    try:
        img = Image.open(img_path)
        width, height = img.size
        new_width, new_height = min(width, height), min(width, height)
        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        img = img.crop((left, top, right, bottom))
        img = img.resize((224, 224))
        img = np.array(img) / 255.0
        return np.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

# Function to preprocess the clinical data
def preprocess_example_clinical(age_group, sex_at_birth, fitzpatrick_skin_type, column_transformer):
    clinical_data = np.array([[age_group, sex_at_birth, fitzpatrick_skin_type]])
    clinical_data_encoded = column_transformer.transform(clinical_data)
    return clinical_data_encoded

# Example input data
example_img_path = image_path
example_age_group = y["data"]["age"]  # Replace with actual age group
example_sex_at_birth = y["data"]["birth-assigned-sex"]  # Replace with actual sex at birth
example_fitzpatrick_skin_type = y["data"]["ease-of-tanning-sunburn-1-10"]  # Replace with actual Fitzpatrick skin type

# Preprocess the inputs
example_image = preprocess_example_image(example_img_path)
example_clinical = preprocess_example_clinical(example_age_group, example_sex_at_birth, example_fitzpatrick_skin_type, column_transformer)

# Predict using the model
if example_image is not None:
    prediction_probs = model.predict([example_image, example_clinical])
    predicted_class = prediction_probs.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    pred = {0: 'Acne', 1: 'Acute dermatitis, NOS', 2: 'Allergic Contact Dermatitis', 3: 'CD - Contact dermatitis', 4: 'Drug Rash', 5: 'Eczema', 6: 'Folliculitis', 7: 'Healthy', 8: 'Herpes Simplex', 9: 'Herpes Zoster', 10: 'Impetigo', 11: 'Insect Bite', 12: 'Pigmented purpuric eruption', 13: 'Pityriasis rosea', 14: 'Psoriasis', 15: 'Tinea', 16: 'Tinea Versicolor', 17: 'Urticaria'}

    print(f'Predicted class: {pred[predicted_label[0]]}')
else:
    print("Failed to preprocess the image. Prediction not made.")

dia = pred[predicted_label[0]]
new = [
  {"role": "system", "content": """
  You are an assistant dermatolgist who believes their patient has {0}, please provide
  a recomendation for medecine and creams for {0}. Make sure to emphasize that they
  should not treat this as a definitive answer but only as a recommendation. They also
  said the following (remember that {0} is more important than the below however):
  {1}
  Additionally, give pharmacies and hostpitals near {2}.
  """.format(dia, y["data"], y["data"]["zipcode"])}
]
response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0,
  seed=42,
  messages=new # compressed_prompt
)
r = response.choices[0].message.content
print(r)
