import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk, os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim import corpora, models

# download nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')

# make output folder
OUT_DIR = "./out"
os.makedirs(OUT_DIR, exist_ok=True)


# ----- 0. Load Dataset -----
# dataset already cleaned
# use description + text for all text analysis
df = pd.read_csv("./out/df_cleaned.csv")
df['combined_text'] = df['description'].astype(str) + " " + df['text'].astype(str)

# token cleaning: lowercase, remove stopwords, lemmatize
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tokens(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return tokens

df['tokens'] = df['combined_text'].apply(clean_tokens)
print(df['tokens'].head())


# ----- 1. Word Frequency -----
# count words and show most common
all_words = [w for tokens in df['tokens'] for w in tokens]
fdist = FreqDist(all_words)
print(fdist.most_common(20))

# save plot of top 20 words by freq
plt.figure()
fdist.plot(20, title="Top 20 words")
plt.savefig(os.path.join(OUT_DIR, "wordfreq_top20.png"))
plt.close()


# ----- 2. Word Cloud -----
# show big picture of words
# easy to see human vs bot language
wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_words))
plt.figure(figsize=(12,6))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Wordcloud")
plt.savefig(os.path.join(OUT_DIR, "wordcloud.png"))
plt.close()


# ----- 3. TF-IDF -----
# convert text into numeric matrix using TF-IDF
# highlights distinctive words for accounts
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['combined_text'])
print("TF-IDF shape:", X_tfidf.shape)
print(vectorizer.get_feature_names_out()[:30])


# ----- 4. Topic Modeling LDA -----
# Latent Dirichlet Allocation finds hidden themes inside text
# emotional/social themes for humans
# neutral/promotional themes for bots or brands
texts = df['tokens']
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=5, random_state=42)

print("LDA topics:")
for idx, topic in lda_model.print_topics(-1):
    print("Topic", idx, ":", topic)


# ----- 5. Unique Words -----
# check vocab diversity
# humans usually have more variety
unique_tokens = set(all_words)
print("Total tokens:", len(all_words))
print("Unique tokens:", len(unique_tokens))
print("Sample unique words:", list(unique_tokens)[:20])


# ----- 6. Sentiment Analysis -----
# calculate sentiment polarity scores for each profile
# distribution shows humans have wider positive/negative spread
# bots/brands stay more around neutral
sia = SentimentIntensityAnalyzer()
df['sentiment'] = df['combined_text'].astype(str).apply(lambda x: sia.polarity_scores(x)['compound'])

print(df[['combined_text','sentiment']].head())

# sentiment distribution plot
plt.figure(figsize=(6,4))
sns.histplot(df['sentiment'], bins=40, kde=True)
plt.title("Sentiment score distribution")
plt.savefig(os.path.join(OUT_DIR, "sentiment_distribution.png"))
plt.close()

# group check by gender type
print(df.groupby('gender')['sentiment'].mean())
