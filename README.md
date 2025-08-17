📑 **Sentiment Analysis of Movie Reviews**

**1. Introduction**

The goal of this project was to build a machine learning model capable of classifying movie reviews as either positive or negative. Using the IMDb Movie Reviews Dataset (50,000 reviews), we implemented a pipeline that preprocesses text, trains multiple models, evaluates their performance, and finally deploys the best model in a Streamlit web application for real-time predictions.

**2. Approach**

=> Dataset: IMDb Movie Reviews Dataset
 (https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

a) Data Preprocessing

Removed HTML tags, punctuation, and non-alphabetic characters.

Converted text to lowercase.

Tokenized sentences and removed stopwords using NLTK.

Applied TF-IDF Vectorization to transform text into numerical feature vectors. Initially, we limited features to 5,000 words (max_features=5000) for efficiency, but also tested with the full vocabulary.

b) Model Training

We trained three traditional ML algorithms well-suited for text classification:

Logistic Regression

Naïve Bayes (MultinomialNB)

Support Vector Machine (LinearSVC)

Each model was trained on 80% of the dataset and evaluated on the remaining 20%.

c) Evaluation

Metrics used: Accuracy and F1-Score (to balance precision and recall).

Results showed that Logistic Regression performed best overall, achieving high accuracy and F1-score while being computationally efficient.

The best model was saved as sentiment_model.pkl, along with the TF-IDF vectorizer.

d) Deployment (Streamlit App)

A Streamlit-based web app was developed (app.py).

Users can input reviews and instantly receive predictions.

The app shows not only the sentiment (positive/negative) but also the confidence percentage, making results more interpretable.

The interface was styled with a cinema-inspired theme (dark background with bluish/purplish gradients, styled buttons, and emojis) for better user engagement.

**3. Challenges**

High-dimensional Data: The IMDb dataset contains a large vocabulary. Using all words significantly increased computation time and memory usage. We addressed this by experimenting with max_features in TF-IDF.

Ambiguous Reviews: Some reviews contained both positive and negative comments, making classification harder. While models achieved good accuracy, handling mixed sentiment remains a challenge.

Model Confidence: SVM models don’t naturally provide probabilities. We solved this by approximating confidence using the decision function or choosing Logistic Regression, which directly supports probability estimates.

Deployment Issues: Ensuring all dependencies were properly listed in requirements.txt was necessary for smooth deployment.

**4. Outcomes**

Built a robust sentiment classifier with accuracy above 97% (Logistic Regression).

Deployed an interactive web app where users can test reviews in real time.

Enhanced UX with confidence scores, custom CSS styling, and explanatory sections.

Demonstrated the full ML pipeline: data preprocessing → training → evaluation → deployment.


**Final Deliverables:**

sentiment_analysis.ipynb → Model training & evaluation.

app.py → Streamlit app.

requirements.txt → Dependencies.


Trained model + vectorizer (sentiment_model.pkl, vectorizer.pkl).
