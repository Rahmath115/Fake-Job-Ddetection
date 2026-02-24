import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Step 1: Unzipping dataset...")

with zipfile.ZipFile("fake_job_postings.csv.zip", 'r') as zip_ref:
    zip_ref.extractall()

print("Dataset unzipped ✅")

print("Step 2: Loading dataset...")

data = pd.read_csv("fake_job_postings.csv")

print("Dataset loaded ✅")

print("Step 3: Cleaning data...")

data = data[['description','fraudulent']]
data = data.dropna()

print("Data cleaned ✅")
data['description'] = data['description'].str.lower()
data['description'] = data['description'].str.replace(r'[^a-zA-Z ]', '', regex=True)
print("Step 4: Converting text to numbers...")

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(data['description'])
y = data['fraudulent']

print("Conversion done ✅")

print("Step 5: Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Split done ✅")

print("Step 6: Training model...")
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

print("🎉 Model trained successfully!")

print("\nStep 7: Evaluating model...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("📊 Model Accuracy:", accuracy)

print("\nStep 8: Testing model...")

real_job = ["We are hiring a software engineer with experience in Python and data analysis."]
fake_job = ["Earn money fast from home. No experience needed. Click now!"]

real_vec = vectorizer.transform(real_job)
fake_vec = vectorizer.transform(fake_job)

real_prediction = model.predict(real_vec)
fake_prediction = model.predict(fake_vec)

if real_prediction[0] == 1:
    print("🚨 REAL job predicted as FAKE")
else:
    print("✅ REAL job predicted as REAL")

if fake_prediction[0] == 1:
    print("🚨 FAKE job predicted as FAKE")
else:
    print("❌ FAKE job predicted as REAL")
import pickle

# Save trained model
pickle.dump(model, open("model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model & Vectorizer Saved Successfully ✅")