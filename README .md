# HAM OR SPAM? — SMS Spam Classifier

A minimalist old-school newspaper/comic-style web application that classifies SMS messages as **HAM** or **SPAM** using **Multinomial Naive Bayes**.

## How it works

```text
SMS message
    ↓
Text preprocessing
    ↓
CountVectorizer
    ↓
Multinomial Naive Bayes
    ↓
HAM / SPAM
```

### Machine Learning pipeline

1. The SMS dataset is cleaned and duplicate messages are removed.
2. Text is converted to lowercase.
3. Words are tokenized.
4. Non-alphanumeric tokens and English stopwords are removed.
5. Porter stemming is applied.
6. `CountVectorizer` converts the processed text into numerical word-count features.
7. `MultinomialNB` is trained on the resulting features.
8. The trained model and vectorizer are saved as:
   - `model.pkl`
   - `vectorizer.pkl`

The Streamlit application loads these saved files and performs predictions on new messages.

## Project structure

```text
SMS-Spam-Classifier/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── email.csv
└── SMS_Spam_Classifier_MultinomialNB_Updated.ipynb
```

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Deployment

This project can be deployed directly from a GitHub repository using Streamlit Community Cloud.

Set the main file to:

```text
app.py
```

The application does not retrain the model during deployment. It loads the already-trained `model.pkl` and `vectorizer.pkl` files.

## Model

The deployed classifier is:

**Multinomial Naive Bayes**

It is a natural choice for the word-count features produced by `CountVectorizer`.

## Author

Asrar Farooq Wani
