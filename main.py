import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Lirik lagu
lyrics = """
[Verse 1]
We don't talk about this back home
Lock me up if I talk about this back home
Mistreated some of us back home
Colonized what's left in the West back home
We sure play pretend back home
Picking scraps of what we used to call home
Cautious of how I use my phone
Fearful of how I set my tone

[Pre-Chorus]
Can only sing these songs in English
I know my chances, yeah, so I'll take the piss
I don't have much time, let me finish
A little space for things to say
That got me

[Chorus]
Oh, hated in the nation
I know I'll be
Oh, hated in the nation
The nation

[Verse 2]
We can't talk about this back home
Tiptoe when you talk about home
I know you get demonized back home
Full speed demolition back home

[Pre-Chorus]
Can only sing these songs in English
Unless I wanna sleep with the fishes
Saudara sedarah menangis
A little space for things to say
That got me

[Chorus]
Oh, hated in the nation
I know I'll be
Oh, hated in the nation
The nation
Oh, hated in the nation
I know I'll be
Oh, hated in the nation
The nation

"""

# Preprocessing
lyrics = lyrics.lower()
tokens = word_tokenize(lyrics)
stop_words = set(stopwords.words('english'))
filtered_words = [word for word in tokens if word.isalpha() and word not in stop_words]

# Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_words))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud Lirik")
plt.show()

# Load NRC Emotion Lexicon (format: word, emotion, association)
nrc = pd.read_csv('/content/nrc.txt', sep='\t', names=['word', 'emotion', 'association'])
nrc = nrc[nrc['association'] == 1]

# Gabungkan lirik dengan emosi
emotion_dict = {}
for word in filtered_words:
    emotions = nrc[nrc['word'] == word]['emotion'].tolist()
    for emotion in emotions:
        if emotion in emotion_dict:
            emotion_dict[emotion] += 1
        else:
            emotion_dict[emotion] = 1

# Visualisasi Emosi
emotion_df = pd.DataFrame(list(emotion_dict.items()), columns=['Emotion', 'Count'])
plt.figure(figsize=(10, 5))
sns.barplot(x='Emotion', y='Count', data=emotion_df, palette='pastel')
plt.title('Distribusi Emosi dalam Lirik Lagu')
plt.xticks(rotation=45)
plt.show()
