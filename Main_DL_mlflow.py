import numpy as np
import pandas as pd
import nltk
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.corpus import reuters
from nltk.corpus import brown
from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
import joblib
from collections import Counter
from textblob import Word 
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, f1_score, recall_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.utils import pad_sequences

# from tensorflow.keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Activation, Dense, Embedding, LSTM, SpatialDropout1D, Dropout, Flatten, GRU, Conv1D, MaxPooling1D, Bidirectional
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests
import re

# MLflow imports
import mlflow           
import mlflow.sklearn   
import mlflow.tensorflow 
import tempfile

# import ktrain
# from ktrain import text

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('gutenberg')
nltk.download('brown')
nltk.download("reuters")
nltk.download('words')

df=pd.read_csv("./data/bbc-text.csv", engine='python', encoding='UTF-8')
df['category'].value_counts()

df.to_csv("bbc-text.csv", index=False)

df['category'].value_counts()


vocabulary_size = 15000
max_text_len = 768
stemmer = SnowballStemmer('english')
stop_words = [word for word in stopwords.words('english') if word not in ["my","haven't","aren't","can","no", "why", "through", "herself", "she", "he", "himself", "you", "you're", "myself", "not", "here", "some", "do", "does", "did", "will", "don't", "doesn't", "didn't", "won't", "should", "should've", "couldn't", "mightn't", "mustn't", "shouldn't", "hadn't", "wasn't", "wouldn't"]]
  
def preprocess_text(text):
  
    text = re.sub('[^a-zA-Z]', ' ', text)

    words = text.lower().split()
 
    words = [stemmer.stem(word) for word in words if not word in stop_words]
   
    cleaned_text = ' '.join(words)
    return cleaned_text

df['cleaned_text'] = df['text'].apply(preprocess_text)

tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['cleaned_text'].values)
le = len(tokenizer.word_index) + 1
print(le)
sequences = tokenizer.texts_to_sequences(df['cleaned_text'].values)
X_DeepLearning = pad_sequences(sequences, maxlen=max_text_len)

mlflow.set_experiment("Document Classification")

# fix for case sensitive labels
df.loc[df['category'] == 'sport' , 'LABEL'] = 0     
df.loc[df['category'] == 'business', 'LABEL'] = 1
df.loc[df['category'] == 'politics' , 'LABEL'] = 2    
df.loc[df['category'] == 'tech', 'LABEL'] = 3              
df.loc[df['category'] == 'entertainment', 'LABEL'] = 4             

labels = to_categorical(df['LABEL'], num_classes=5)
XX_train, XX_test, y_train, y_test = train_test_split(X_DeepLearning , labels, test_size=0.25, random_state=42)
print((XX_train.shape, y_train.shape, XX_test.shape, y_test.shape))


epochs = 25
emb_dim = 256
batch_size = 50

mlflow.tensorflow.autolog()

with mlflow.start_run(run_name="04_LSTM_DL"):
    mlflow.set_tag("source_data", "bbc_v1_seed42")

    model_lstm1 = Sequential()
    model_lstm1.add(Embedding(vocabulary_size,emb_dim, input_length=X_DeepLearning.shape[1]))
    model_lstm1.add(SpatialDropout1D(0.8))                                             
    model_lstm1.add(Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5)))                 
    model_lstm1.add(Dropout(0.5))
    model_lstm1.add(Flatten())
    model_lstm1.add(Dense(64, activation='relu'))
    model_lstm1.add(Dropout(0.5))
    model_lstm1.add(Dense(5, activation='softmax'))
    model_lstm1.compile(optimizer=tf.optimizers.Adam(),loss='categorical_crossentropy', metrics=['acc']) 

    model_lstm1.build(input_shape=(None, X_DeepLearning.shape[1]))

    print(model_lstm1.summary())  

    # call back
    checkpoint_callback = ModelCheckpoint(filepath="lastm-1-layer-best_model.h5", save_best_only=True, monitor="val_acc", mode="max", verbose=1)
    early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=10, verbose=1, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, verbose=1, mode="min", min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback]

    # train
    history_lstm1 = model_lstm1.fit(XX_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(XX_test,y_test), callbacks=callbacks)

    # eval
    results_1 = model_lstm1.evaluate(XX_test, y_test, verbose=False)
    print(f'Test results - Loss: {results_1[0]} - Accuracy: {100*results_1[1]}%')
    mlflow.log_metric("final_test_accuracy", results_1[1])

    acc = history_lstm1.history['acc']                        
    val_acc = history_lstm1.history['val_acc']
    loss = history_lstm1.history['loss']
    val_loss = history_lstm1.history['val_loss']
    plt.plot( acc, 'go', label='Train accuracy')
    plt.plot( val_acc, 'g', label='Validate accuracy')
    plt.title('Train and validate accuracy')
    plt.legend()         

    with tempfile.NamedTemporaryFile(prefix="lstm_accuracy", suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="visualizations")
        plt.close()                    

    plt.figure()
    plt.plot( loss, 'go', label='Train loss')
    plt.plot( val_loss, 'g', label='Validate loss')
    plt.title('Train and validate loss')
    plt.legend()
    plt.show()

    with tempfile.NamedTemporaryFile(prefix="lstm_loss", suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="visualizations")
        plt.close()

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact('tokenizer.pickle', artifact_path="model_assets")


    ## Predict
    sample_text = "howard hits back at mongrel jibe michael howard has said a claim by peter hain"
    cleaned_sample = preprocess_text(sample_text)
    seq = tokenizer.texts_to_sequences([cleaned_sample])
    padded = pad_sequences(seq, maxlen=max_text_len)
    pred = model_lstm1.predict(padded)
    labels_map = {0: 'Sport', 1: 'Business', 2: 'Politics', 3: 'Tech', 4: 'Entertainment'}
    predicted_label = labels_map[np.argmax(pred)]
    
    print(f"Sample Prediction: {predicted_label}")
    mlflow.log_param("sample_text_prediction", predicted_label)


    # confusion matrix and classification report    
    print("Confusion Matrix:")
    y_pred_probs = model_lstm1.predict(XX_test)
    
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sport', 'Business', 'Politics', 'Tech', 'Ent'], 
                yticklabels=['Sport', 'Business', 'Politics', 'Tech', 'Ent'])
    plt.title('Confusion Matrix: LSTM')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    with tempfile.NamedTemporaryFile(prefix="lstm_cm", suffix=".png", delete=False) as tmp:
        plt.savefig(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="visualizations")
        plt.close()

    print("Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=['Sport', 'Business', 'Politics', 'Tech', 'Ent']))
    
    report = classification_report(y_true_classes, y_pred_classes, target_names=['Sport', 'Business', 'Politics', 'Tech', 'Ent'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    with tempfile.NamedTemporaryFile(prefix="lstm_report", suffix=".csv", mode='w+', delete=False) as tmp:
        report_df.to_csv(tmp.name)
        mlflow.log_artifact(tmp.name, artifact_path="reports")