# NLP-Greek-Language-Proficiency-Classification

## Project Overview
This repository is dedicated to the NLP Greek Language Proficiency Classification project, which focuses on classifying Greek texts into CEFR (Common European Framework of Reference for Languages) levels using an SVM model. This project employs Natural Language Processing and machine learning techniques to assess language skills efficiently.

## Motivation
The goal of this project is to provide an automated tool for the classification of Greek language proficiency, aiding educational institutions and learners by delivering swift and accurate assessments of language capabilities.

## Technologies Used
- Python
- Scikit-Learn
- NLTK
- spaCy
- Gensim for Word2Vec
- SMOTE for handling class imbalance

## Features
- Advanced text preprocessing: tokenization, lemmatization, and removal of stop words.
- Feature extraction with TF-IDF, n-grams, and Word2Vec.
- Implementation of an SVM classifier fine-tuned using GridSearchCV.

## Installation
To set up this project for use or development, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/spymavro/NLP-Greek-Language-Proficiency-Classification.git
   cd NLP-Greek-Language-Proficiency-Classification
2. **Install the required Python packages**:

- Ensure you have Python installed on your system. If not, download and install it from [python.org](https://www.python.org/downloads/).
- It's recommended to create a virtual environment to keep dependencies required by different projects separate and to avoid conflicts:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows use `venv\Scripts\activate`
-Install the required packages:
    pip install scikit-learn nltk spacy gensim
-After installing spacy, you may need to download a specific language model. For Greek, you can use:
    python -m spacy download el_core_news_sm
3. **Proceed with project setup and usage as required**:
### Explanation:
- **Step 1**: Cloning the repository is straightforward; make sure to replace `spymavro` with your actual GitHub username.
- **Step 2**: This step covers:
  - Checking for Python installation.
  - Setting up a virtual environment, which is optional but recommended.
  - Direct installation of each required Python package using `pip`.
  - Additional steps for setting up `spacy` with the Greek language model are included since it's a common requirement for projects involving NLP with Greek text.

This approach ensures that users have a clear, step-by-step guide to setting up the project environment, even without a `requirements.txt` file.

## Usage
Run the SVM classification model with the following command:
python svm_classifier.py

Note: This repository only includes the code for the SVM model, which I developed. Other machine learning models used in this project by my colleagues are not shared here.

## All Rights Reserved
This project and all its contents are copyrighted and cannot be copied, modified, or distributed without express permission from the creator.

## Contact
For permissions or inquiries, please contact me at spyros.mauromatis@gmail.com



