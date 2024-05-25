# -*- coding: utf-8 -*-

# -- Sheet --

# # Fake and real news dataset


# Fake news spread through social media has become a serious problem. Can we use this data set to make an algorithm able to determine if an article is fake news or not ?
# 
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# 
# This dataset comprises approximately 40,000 articles including both fake and authentic news (around 20,000 each). Our goal is to find and train a model to accurately predict whether a given piece of news is genuine or fabricated.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from datetime import datetime

# ## Import Dataset


true = pd.read_csv('/data/notebook_files/True.csv')
fake = pd.read_csv('/data/notebook_files/Fake.csv')

print(f'Rows and columns of True news: {true.shape} \nRows and columns of False news: {fake.shape}')

true.head()

fake.head()

import textwrap
print(textwrap.fill('True news:   '+true.text[0], 120))

print(textwrap.fill('Fake news:   '+ fake.text[0], 120))
# there are many @ in the fake news 

# ## Visualization of the Raw Data


# ### 1. Add a column for True or Fake and a column for text length, then Union two sets together  


true['label'] = 1
fake['label'] = 0

true['text_length'] = true['text'].apply(len)
fake['text_length'] = fake['text'].apply(len)

news = pd.concat([true, fake])
news.shape

news.info()

# ### 2. Check N/A


news.isna().sum() 

# #### 2.1 the date value of some rows are not dates


# some values in the date column was url or an article -- delete these rows
indices_to_drop = news[news['date'].str.len()>30].index #the indices of rows to drop
news = news.drop(indices_to_drop)
news.shape

# ### 3. Summary Statistics


plt.figure(figsize = [4, 4], clear = True, facecolor = 'white')
sns.barplot(x = news['label'].value_counts().index,
            y = news['label'].value_counts(),
            saturation = 0.7).set(title = 'Count of the news by True - 1/Fake - 0)')

# The data is balanced between true and fake news


# #### 3.1 Count of news by date &  Oldest and Latest date


news.groupby('label').date.nunique()

news['date'] = pd.to_datetime(news['date'])

news.groupby('label').date.max(), \
news.groupby('label').date.min() 

# #### 3.2 Count of news by subject


news.groupby('label').subject.value_counts()

plt.figure(figsize = [12, 5], clear = True, facecolor = 'white')
sns.barplot(x = news['subject'].value_counts().index,
            y = news['subject'].value_counts(),
            saturation = 0.6).set(title = 'Count of the news by Subject')

fig = px.pie(news, names = 'subject', title = 'Count of the news by Subject', hole = 0.2,
            width = 800, height = 400, color_discrete_sequence = px.colors.sequential.YlGnBu)
fig.update_layout(title_x = 0.5, title_font = dict(size = 15), uniformtext_minsize = 25)
fig.show()

# #### 3.3 Count of tokens by subject -- Take out the @ sign in the fake news could make the articles shorter
# Token: a single unit of text, which could be a word, a punctuation mark, or any other meaningful element derived from the text.


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Box(y=list(fake['text_length']), name='Fake'))
fig.add_trace(go.Box(y=list(true['text_length']), name = 'Real'))

fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'title': 'Box plot',
})
fig.show()

# #### 3.4 Count of @ by subject -- Take out the @ in the model analysis


# create a new column for the count of "@"
news['count_At'] = news['text'].apply(lambda x: x.count('@'))
news['count_At'].describe()

# the count of "@" by Real/Fake news
news[news.count_At > 0].groupby('label').count_At.count()

plt.hist(news[news.label==1]['count_At'], bins=range(1, news[news.label==1]['count_At'].max()), align='left')
plt.xlabel('Count of @')
plt.ylabel('Frequency')
plt.title('Distribution of @ Counts in Real News')
plt.show()

# Create a histogram to show the distribution of '@' counts
plt.hist(news[news.label==0]['count_At'], bins=range(1, news[news.label==0]['count_At'].max()), align='left')
plt.xlabel('Count of @')
plt.ylabel('Frequency')
plt.title('Distribution of @ Counts in Fake News')
plt.show()

# #### 3.5 Wordcloud


from wordcloud import WordCloud

wc = WordCloud(background_color = 'white', width = 800, height = 400,
               contour_width = 0, contour_color = 'red', max_words = 1000,
               scale = 1, collocations = False, repeat = True, min_font_size = 1)

# Real news wordcloud

text = " ".join(i for i in true.text)
wc.generate(text)

plt.figure(figsize = [10, 7])
plt.imshow(wc)
plt.show

# Fake news wordcloud

text = " ".join(i for i in fake.text)
wc.generate(text)

plt.figure(figsize = [10, 7])
plt.imshow(wc)
plt.show

# ## Data Preprocessing


# ### 1. Convert all text to lower case


news['title'] = news['title'].str.lower()
news['text'] = news['text'].str.lower()
news['subject'] = news['subject'].str.lower()
news.head(2)

# ### 2. String cleaning 
# - Remove URLs, numbers and punctuation
# - Normalize unicode characters like U+0041 for the letter "A", and U+1F60A for the emoji "😊"


# urls
news['text'] = news['text'].str.replace('https?:\/\/.*[\r\n]*', '', regex=False)

# numbers 
news['text'] = news['text'].str.replace('\d+', '', regex=False)

# punctuation
news['text'] = news['text'].str.replace('[^\w\s]', '', regex=False)

#unicode
news['text'] = news['text'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# ### 3. Remove stop words 
# 
# - Stop words include words like "and," "the," "is," "in," "at," "on," etc., which are essential for the structure of sentences but might not contribute much to the understanding or interpretation of the content when analyzing the text.


import nltk
nltk.download('stopwords')
nltk.download('punkt') 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize   # performs tokenization, which is the process of breaking down a text into individual words.

# use the stopwords class from nltk package
stop_words = set(stopwords.words('english'))

# define a function to remove the stop words 
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Apply the function on the column "text"
news['text'] = news['text'].apply(remove_stopwords)

# ## Modeling - Clustering on news subject


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# ### All news subject clustering 


# The Adjusted Rand Index (ARI) is a measure of the similarity between two data clusterings. 
# - It accounts for chance agreement between clusterings by comparing the proportion of agreements to the expected agreement under independence. 
# - The Adjusted Rand Index ranges from -1 to 1. A score of 1 indicates perfect agreement between the two clusterings, a score of 0 indicates random agreement, and negative scores indicate disagreement worse than random.
# 
# The Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
# - A score close to +1 indicates that the sample is far away from neighboring clusters, meaning it is well-clustered.
# - A score close to 0 indicates that the sample is close to the decision boundary between two neighboring clusters.
# - A score close to -1 indicates that the sample may have been assigned to the wrong cluster.


X_text = news['text'].values
subjects = news['subject'].values

# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_tfidf = vectorizer.fit_transform(X_text)

num_clusters = [3,4,5,6,7,8,9]

for num in num_clusters:
    # Apply K-means clustering
    # n_init is the number of times the K-means algorithm will be run with different initial centroids. The final result is the best clustering result (lowest inertia or within-cluster sum of squares) among all the runs. A larger n_init value increases the likelihood of finding a better overall clustering solution.
    kmeans = KMeans(n_clusters = num, random_state = 42, n_init = 20)

    clusters = kmeans.fit_predict(X_text_tfidf)

    # Add cluster labels to the DataFrame
    news['cluster'] = clusters

    # compare subject value and cluster
    ari_score = adjusted_rand_score(news['subject'], news['cluster'])
    
    print(f'Number of clusters is {num}.\n Adjusted Rand Index: {ari_score}')

# ### Real news clustering


real_news = news.loc[news['label']==1, :].copy()
X_text_real = real_news['title'].values

# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_real_tfidf = vectorizer.fit_transform(X_text_real)

num_clusters = [2, 3, 4, 5, 6]

for num in num_clusters:
    # Apply K-means clustering
    # n_init is the number of times the K-means algorithm will be run with different initial centroids. The final result is the best clustering result (lowest inertia or within-cluster sum of squares) among all the runs. A larger n_init value increases the likelihood of finding a better overall clustering solution.
    kmeans = KMeans(n_clusters = num, random_state = 42, n_init = 20)

    clusters = kmeans.fit_predict(X_text_real_tfidf)

    # Add cluster labels to the DataFrame
    real_news.loc[:, 'cluster'] = clusters

    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X_text_real_tfidf, clusters)

    # compare subject value and cluster
    ari_score = adjusted_rand_score(real_news['subject'], real_news['cluster'])
    
    print(f'For real news: \n Number of clusters is {num}.\n Adjusted Rand Index: {ari_score}.\n Silhouette Score: {silhouette_avg}')

real_news = news.loc[news['label']==1, :].copy()
X_text_real = real_news['text'].values

# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_real_tfidf = vectorizer.fit_transform(X_text_real)

num_clusters = [2, 3, 4, 5, 6]

for num in num_clusters:
    # Apply K-means clustering
    # n_init is the number of times the K-means algorithm will be run with different initial centroids. The final result is the best clustering result (lowest inertia or within-cluster sum of squares) among all the runs. A larger n_init value increases the likelihood of finding a better overall clustering solution.
    kmeans = KMeans(n_clusters = num, random_state = 42, n_init = 20)

    clusters = kmeans.fit_predict(X_text_real_tfidf)

    # Add cluster labels to the DataFrame
    real_news.loc[:, 'cluster'] = clusters

    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X_text_real_tfidf, clusters)

    # compare subject value and cluster
    ari_score = adjusted_rand_score(real_news['subject'], real_news['cluster'])
    
    print(f'For real news: \n Number of clusters is {num}.\n Adjusted Rand Index: {ari_score}.\n Silhouette Score: {silhouette_avg}')

# ### Fake news clustering


fake_news = news.loc[news['label']==0, :].copy()
X_text_fake = fake_news['title'].values

# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_fake_tfidf = vectorizer.fit_transform(X_text_fake)

num_clusters = [2, 3, 4, 5, 6]

for num in num_clusters:
    # Apply K-means clustering
    # n_init is the number of times the K-means algorithm will be run with different initial centroids. The final result is the best clustering result (lowest inertia or within-cluster sum of squares) among all the runs. A larger n_init value increases the likelihood of finding a better overall clustering solution.
    kmeans = KMeans(n_clusters = num, random_state = 42, n_init = 20)

    clusters = kmeans.fit_predict(X_text_fake_tfidf)

    # Add cluster labels to the DataFrame
    fake_news.loc[:, 'cluster'] = clusters

    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X_text_fake_tfidf, clusters)

    # compare subject value and cluster
    ari_score = adjusted_rand_score(fake_news['subject'], fake_news['cluster'])
    
    print(f'For fake news: \n Number of clusters is {num}.\n Adjusted Rand Index: {ari_score}.\n Silhouette Score: {silhouette_avg}')

fake_news = news.loc[news['label']==0, :].copy()
X_text_fake = fake_news['text'].values

# Preprocess and vectorize the text
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_text_fake_tfidf = vectorizer.fit_transform(X_text_fake)

num_clusters = [2, 3, 4, 5, 6]

for num in num_clusters:
    # Apply K-means clustering
    # n_init is the number of times the K-means algorithm will be run with different initial centroids. The final result is the best clustering result (lowest inertia or within-cluster sum of squares) among all the runs. A larger n_init value increases the likelihood of finding a better overall clustering solution.
    kmeans = KMeans(n_clusters = num, random_state = 42, n_init = 20)

    clusters = kmeans.fit_predict(X_text_fake_tfidf)

    # Add cluster labels to the DataFrame
    fake_news.loc[:, 'cluster'] = clusters

    # Compute Silhouette Score
    silhouette_avg = silhouette_score(X_text_fake_tfidf, clusters)

    # compare subject value and cluster
    ari_score = adjusted_rand_score(fake_news['subject'], fake_news['cluster'])
    
    print(f'For fake news: \n Number of clusters is {num}.\n Adjusted Rand Index: {ari_score}.\n Silhouette Score: {silhouette_avg}')

# There could be several reasons why the clustering performance appears poor:
# 
# Sparse Text Data: Text data tends to be high-dimensional and sparse, especially when represented using techniques like TF-IDF. In high-dimensional spaces, traditional distance-based clustering algorithms like K-means may struggle to find meaningful clusters, leading to suboptimal results.
# 
# Lack of Inherent Structure: Real news articles may not exhibit clear and distinct clusters based solely on their text content. Unlike datasets with well-defined clusters, such as customer segmentation data or image data, the natural structure of news articles might be more complex and less amenable to clustering.
# 
# Noise and Variability: News articles cover a wide range of topics and writing styles, leading to a high degree of variability within the dataset. This variability can introduce noise and make it challenging for clustering algorithms to identify meaningful patterns.
# 
# Curse of Dimensionality: In high-dimensional spaces, the distance between points becomes less meaningful, making it harder for clustering algorithms to discern clusters accurately. This issue is exacerbated when using text data with a large number of features (e.g., TF-IDF vectors with many unique terms).
# 
# Given these challenges, it's essential to consider alternative approaches or preprocessing steps to improve clustering performance:
# 
# Dimensionality Reduction: Use techniques like principal component analysis (PCA) or t-distributed stochastic neighbor embedding (t-SNE) to reduce the dimensionality of the TF-IDF vectors before clustering.
# 
# Alternative Clustering Algorithms: Experiment with other clustering algorithms that are more robust to high-dimensional and sparse data, such as spectral clustering or hierarchical clustering.
# 
# Feature Engineering: Explore different text representations or feature engineering techniques to capture more meaningful patterns in the text data, such as word embeddings or topic modeling.
# 
# Domain Knowledge: Incorporate domain knowledge or additional metadata (e.g., publication date, author information) to enrich the clustering process and guide the interpretation of the results.


# ## Modeling - Real or Fake Classification 


# ### 1. "Shallow": Multinomial Naive Bayes classifier
# 
# Multinomial Naive Bayes is a probabilistic classification algorithm that is widely used for text classification tasks, such as spam filtering, sentiment analysis, and topic categorization. It is a variant of the Naive Bayes algorithm, which is based on Bayes' theorem of probability
# 
# In Multinomial Naive Bayes, the underlying probability distribution used for modeling the data is the multinomial distribution. This distribution is suitable for discrete data, where each feature represents the count or frequency of a term in a document.
# 
# Multinomial Naive Bayes is particularly well-suited for text classification tasks where the input features are the word frequencies or term frequencies in a document. Each document is represented as a vector of word counts, and the classifier predicts the category or class of the document.


# #### 1.1 Prep: Split data into training and testing sets and Vectorize text data


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(news['text'],  news['label'], test_size=0.2, random_state=42)

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# #### 1.2 Train the model


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Make predictions on the test set
predictions = clf.predict(X_test_tfidf)

# #### 1.3 Evaluate the model


print('Accuracy:', accuracy_score(y_test, predictions), '\n')
print('\nClassification Report:\n', classification_report(y_test, predictions), '\n')

cm = confusion_matrix(y_test, predictions)
print('\nConfusion Matrix:\n', cm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# - Cross validation
# 
# a resampling procedure used to assess the performance of a machine learning model. The dataset is divided into several subsets, and the model is trained and evaluated multiple times on different combinations of training and validation sets.


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(clf, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores:", cv_scores)
print("Mean Cross-Validation Accuracy:", cv_scores.mean())

# the results indicate that the Multinomial Naive Bayes model is performing consistently well across different folds, with accuracy scores ranging from approximately 94.57% to 95.21%. 


# ### 2. "Deep": LSTM (Long Short-term Memory)
# 
# Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture designed to address the vanishing gradient problem associated with traditional RNNs. LSTMs are particularly effective for modeling sequences and have found widespread use in natural language processing (NLP) tasks, including text classification.
# 
# In text classification, LSTMs can be used to model sequential dependencies in sentences. The network processes the input text word by word, capturing contextual information and learning to make predictions based on the entire sequence.


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import GridSearchCV

X = news['text'].values
y = news['label'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize and pad sequences
max_words = 5000  
max_len = 200 
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

embedding_dim = 50 

# #### Model 1


# Create the LSTM model
model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)) 

# Adjust based on LSTM units
# the term "units" refers to the number of memory cells or processing units within the LSTM layer. 
# Each LSTM unit contains a set of components, including memory cells and gating mechanisms, allowing the network to capture and control the flow of information over sequential data.
model.add(LSTM(units=100))  

# Binary classification - units=1 is common for this.
# sigmoid maps any real-valued number to the range between 0 and 1. 
model.add(Dense(units=1, activation='sigmoid'))  

# Compile the model
# Adam (short for Adaptive Moment Estimation) is an optimization algorithm that combines the benefits of two other popular optimization algorithms: RMSprop and Momentum. 
# It's designed to provide adaptive learning rates for each parameter, adjusting the learning rates during training based on the historical gradients of each parameter.
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate on the test set
accuracy = model.evaluate(X_test_pad, y_test)[1]
print("Test Set Accuracy:", accuracy)

# Model 1
# The model is compiled using the Adam optimizer and binary cross-entropy loss, with accuracy as the evaluation metric. During training for 5 epochs with a batch size of 32, the model achieves increasingly higher accuracy on both the training and validation sets. 
# The training history shows decreasing loss and increasing accuracy over epochs, indicating successful learning and generalization. The final accuracy on the validation set is 98.48%, suggesting good performance in classifying news articles as real or fake.


# #### Model 2


embedding_vector_features=150
model2 = Sequential()
model2.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
model2.add(Bidirectional(LSTM(200))) 
model2.add(Dropout(0.2))
model2.add(Dense(1,activation='sigmoid'))
model2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model2.summary())

model2.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate on the test set
accuracy = model2.evaluate(X_test_pad, y_test)[1]
print("Test Set Accuracy:", accuracy)






