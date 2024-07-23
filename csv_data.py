import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data from CSV into a DataFrame (assuming 'data.csv' contains 'question' and 'answer' columns)
df = pd.read_csv('train.csv')

# Preprocess the data (tokenization, cleaning, etc.)
# For simplicity, let's assume 'question' is the column containing text data and 'answer' is the label
questions = df['question'].values
answers = df['answer'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(questions, answers, test_size=0.2, random_state=42)

# Create a pipeline that combines a CountVectorizer with a MultinomialNB classifier
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model to a file
joblib.dump(model, 'model.joblib')


#api code
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Define FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('model.joblib')

# Define request body schema using Pydantic BaseModel
class QuestionInput(BaseModel):
    question: str

# Define API endpoint for prediction
@app.post("/predict")
def predict_answer(question_input: QuestionInput):
    # Use the loaded model to make predictions
    answer = model.predict([question_input.question])[0]
    return {"question": question_input.question, "answer": answer}

