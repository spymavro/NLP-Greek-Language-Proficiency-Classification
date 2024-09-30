# NLP-Greek-Language-Proficiency-Classification

## Project Overview
This repository is dedicated to the NLP Greek Language Proficiency Classification project, which focuses on classifying Greek texts into CEFR (Common European Framework of Reference for Languages) levels using three different machine learning models: Support Vector Machine (SVM), Multinomial Naive Bayes, and Recurrent Neural Networks (RNN). This project employs Natural Language Processing and machine learning techniques to assess language skills efficiently.

**Note**: This repository only includes the code for the SVM model, which I developed. Other machine learning models used in this project by my colleagues are not shared here.

## Motivation
The goal of this project is to provide an automated tool for the classification of Greek language proficiency, aiding educational institutions and learners by delivering swift and accurate assessments of language capabilities.

## Dataset
The dataset used in this project was generated from two main sources:

1. **329 authentic texts** from the [Greek Language Center's website](https://www.greek-language.gr/certification/dbs/teachers/index.html) (CEFR levels A1 to C2). These texts are categorized by proficiency levels and can be accessed publicly.
2. **711 generated texts** using **ChatGPT-4o** under specific prompts. These texts were designed to simulate various language proficiency levels (A1 to C2), ensuring a balanced and diverse dataset for training the machine learning models.

### Dataset Versions:
Due to the dataset size and limitations, we organized the texts into two versions:

- **Full CEFR Levels Dataset**: Texts classified into six levels (A1, A2, B1, B2, C1, C2).
- **Condensed Three Levels Dataset**: Texts merged into three broader categories: beginner (A1 + A2), intermediate (B1 + B2), and advanced (C1 + C2).

This dual-source and dual-structure approach enabled better training for machine learning models across different proficiency levels.

**Note**: Due to privacy and licensing reasons, the dataset is not included in this repository.

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

## Results
We evaluated the performance of three models: Support Vector Machine (SVM), Multinomial Naive Bayes, and a Recurrent Neural Network (RNN). The results showed the following:

1. **SVM Model**:
   - **Test Accuracy**: 83.65%
   - **Training Accuracy**: 87.71%
   - **Best Parameters**: `C=10, gamma=0.01`
   - The SVM model performed well overall, but there was some confusion between intermediate and advanced proficiency levels. The model demonstrated strong precision, recall, and F1-scores, particularly for the beginner category.
   - For the **Condensed Three Levels Dataset**, the model performed best, whereas with the **Full CEFR Levels Dataset**, it showed overfitting (test accuracy: 54.32%).

2. **Multinomial Naive Bayes**:
   - **Test Accuracy**: 84.62%
   - **Training Accuracy**: 88.31%
   - Naive Bayes showed slightly better performance than SVM on the test set. It was effective at identifying beginner texts but had some difficulty distinguishing between intermediate and advanced levels.

3. **RNN Model**:
   - **Test Accuracy**: 73%
   - **Training Accuracy**: 93%
   - The RNN model exhibited strong performance on the training set but suffered from overfitting, resulting in a lower test accuracy. The model struggled most with intermediate and advanced level distinctions.

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
- Install the required packages:
  ```bash
  pip install scikit-learn nltk spacy gensim
- After installing spacy, you may need to download a specific language model. For Greek, you can use:
  ```bash
  python -m spacy download el_core_news_sm

3. **Proceed with project setup and usage as required**:
### Explanation:
- **Step 1**: Cloning the repository is straightforward; make sure to replace `spymavro` with your actual GitHub username.
- **Step 2**: This step covers:
  - Checking for Python installation.
  - Setting up a virtual environment, which is optional but recommended.
  - Direct installation of each required Python package using `pip`.
  - Additional steps for setting up `spacy` with the Greek language model are included since it's a common requirement for projects involving NLP with Greek text.

## Usage
- Run the SVM classification model with the following command: 
  ```bash
  python svm_classifier.py 

**Note**: This repository only includes the code for the SVM model, which I developed. Other machine learning models used in this project by my colleagues are not shared here.

## All Rights Reserved
This project and all its contents are copyrighted and cannot be copied, modified, or distributed without express permission from the creator.

## Contact
For permissions or inquiries, please contact me at spyros.mauromatis@gmail.com




