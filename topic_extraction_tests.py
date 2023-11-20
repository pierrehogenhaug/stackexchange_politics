from bs4 import BeautifulSoup
from gensim.models import CoherenceModel
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim.corpora as corpora
import gensim
import nltk
import numpy as np
import pandas as pd
import re
import wandb

wandb.login(key="28e0a54f934e056ba846e10f3460b100aa61283c")


# Read the data
df_comments1 = pd.read_pickle('./pickle_dataframes/comments1.pkl')
df_comments2 = pd.read_pickle('./pickle_dataframes/comments2.pkl')
df_comments = pd.concat([df_comments1,df_comments2])
df_comments.reset_index(drop=True, inplace=True)

df_posts1 = pd.read_pickle('./pickle_dataframes/posts1.pkl')
df_posts2 = pd.read_pickle('./pickle_dataframes/posts2.pkl')
df_posts3 = pd.read_pickle('./pickle_dataframes/posts3.pkl')
df_posts = pd.concat([df_posts1, df_posts2, df_posts3])
df_posts.reset_index(drop=True, inplace=True)

df_postlinks = pd.read_pickle('./pickle_dataframes/posts_links.pkl')
df_tags = pd.read_pickle('./pickle_dataframes/tags.pkl')
df_users = pd.read_pickle('./pickle_dataframes/users.pkl')

# Modify preprocess_text function
def preprocess_text(text, remove_stopwords=False, use_lemmatize=True, use_stemmer=False):
    # Remove non-alphanumeric characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    words = text.split()
    if remove_stopwords:
        words = [word for word in words if word not in stopwords.words('english')]
    if use_lemmatize:
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
    elif use_stemmer:  # Apply stemming only if use_stemmer is True
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    text = ' '.join(words)
    
    return text

# Define apply_lda_and_log function with run_name parameter
def apply_topic_modeling_and_log(df, remove_stopwords, use_lemmatize, use_stemmer, tags_weighting, run_name, ngram_range=(1, 1), max_features=1000):
    # Start a new WandB run with the specified name
    wandb.init(project="stackexchange_politics", entity="s223730", name=run_name)

    # Preprocess Title, Body, and Tags
    df['Title'] = df['Title'].apply(lambda x: preprocess_text(x, remove_stopwords, use_lemmatize, use_stemmer))
    df['Body'] = df['Body'].apply(lambda x: preprocess_text(x, remove_stopwords, use_lemmatize, use_stemmer))
    df['Tags'] = df['Tags'].apply(lambda x: preprocess_text(x, remove_stopwords, use_lemmatize, use_stemmer))


    # Combine Title, Body, and Tags with specified weight for Tags
    # We Keep the original order (title, body, tags) as it reflects the natural flow of information
    df['CombinedText'] = df['Title'] + ' ' + df['Body'] + ' ' + (df['Tags'] * tags_weighting)

    # Create a Dictionary and Corpus needed for Topic Modeling
    words = [doc.split() for doc in df['CombinedText']]
    id2word = corpora.Dictionary(words)
    corpus = [id2word.doc2bow(text) for text in words]

    # Apply TF-IDF with the specified max_features
    # ngram_range=(1, 2) for bi-grams, (1, 3) for tri-grams, and (2, 2) for only bi-grams
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['CombinedText'])

    # Apply LDA and NMF for different numbers of topics
    for n_topics in [5, 10, 15, 20]:
        
        # LDA
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(tfidf_matrix)

        # Calculate Coherence Score
        lda_gensim = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=n_topics, random_state=0)
        coherence_model_lda = CoherenceModel(model=lda_gensim, texts=words, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        # Log Coherence and Perplexity Score
        wandb.log({"coherence_score": coherence_lda, "perplexity_score": lda.perplexity(tfidf_matrix)})
        
        # Extract and log the top words for each topic as a table
        feature_names = tfidf_vectorizer.get_feature_names_out()
        top_words_data = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            top_words_data.append([f"Topic {topic_idx}"] + top_words)

        # Create a WandB Table with top words data
        columns = ["Topic"] + [f"Word {i+1}" for i in range(10)]
        top_words_table = wandb.Table(data=top_words_data, columns=columns)
        
        # Log the table to WandB
        wandb.log({f"n_topics_{n_topics}_cleaned_{str(remove_stopwords)}_lemmatize_{str(use_lemmatize)}_weight_{tags_weighting}": top_words_table})

        # NMF
        nmf_model = NMF(n_components=n_topics, random_state=0)
        nmf_W = nmf_model.fit_transform(tfidf_matrix)
        nmf_H = nmf_model.components_

        # Log the top words for each topic for NMF
        nmf_top_words_data = []
        for topic_idx, topic in enumerate(nmf_H):
            top_words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            nmf_top_words_data.append([f"Topic {topic_idx}"] + top_words)

        nmf_top_words_table = wandb.Table(data=nmf_top_words_data, columns=columns)
        wandb.log({f"nmf_n_topics_{n_topics}": nmf_top_words_table})

    # Close WandB run
    wandb.finish()





def main():
    apply_topic_modeling_and_log(df_posts[df_posts['PostTypeId'] == 1], 
                                remove_stopwords=False, 
                                use_lemmatize=False, 
                                use_stemmer=False,
                                tags_weighting=1, 
                                run_name="MaxFeatures_500",
                                ngram_range=(1, 1),
                                max_features=500)


if __name__ == '__main__':
    main()