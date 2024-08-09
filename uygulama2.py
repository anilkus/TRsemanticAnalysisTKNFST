import pandas as pd
from transformers import pipeline


url = 'https://raw.githubusercontent.com/anilkus/TRsemanticAnalysisTKNFST/main/e-ticaret_urun_yorumlari.csv'
df = pd.read_csv(url, sep=';')
df.columns = ['Metin', 'Durum']

# Duygu analizi modeli ve pipeline oluşturun
classifier = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

# Veri setindeki her yorum için duygu analizini gerçekleştirin
# 'text' yerine 'Metin' kullanarak doğru sütuna erişin
df['predicted_label'] = df['Metin'].apply(lambda x: classifier(x)[0]['label'])
df['confidence'] = df['Metin'].apply(lambda x: classifier(x)[0]['score'])

# Sonuçları görüntüleyin
print(df.head())

# Manuel yorum ekleme
manuel_yorumlar = [
    "Aslında güzel görünüyor ama içini açınca hayal kırıklığına uğradım.",
    "Ürün beklentilerimin altında, hayal kırıklığı yaşadım.",
    "Fiyat/performans oranı oldukça iyi, memnun kaldım."
]

# Manuel yorumlar için duygu analizini gerçekleştirin
manuel_df = pd.DataFrame(manuel_yorumlar, columns=['Metin'])
manuel_df['predicted_label'] = manuel_df['Metin'].apply(lambda x: classifier(x)[0]['label'])
manuel_df['confidence'] = manuel_df['Metin'].apply(lambda x: classifier(x)[0]['score'])

# Sonuçları görüntüleyin
print("Veri setinden analiz edilen yorumlar:")
print(df.head())

print("\nManuel yorumlar için duygu analizi sonuçları:")
print(manuel_df)

#özetleyip analiz etme
# Manuel yorum ekleme
manuel_yorumlar = [
    "ürün geldiğinde şarjı vardı. ilk lullanım öncesi 10 saat kadar şarjda kaldı yaklaşık 30 dk kadar kesintisiz kullandım vebhala şarjı iyi seviyedeydi. 30 dk kullanım sürecinde ısınmadı. ben saç traşı oldum. herhangi bir takılma-çekme vs sorun yaşamadım (saçlarım seyrel ve ince telli. uzunluğu 3-4 mm kadardı, taraksız traş oldum) bu performansı devam ederse mükemmel bir ürün diyebilirim."
]
# Convert list of strings to a single string
text = " ".join(manuel_yorumlar) 
 #Preprocessing   

#text = re.sub(r'\[[0-9]*\]', ' ', text)
#text = re.sub(r'\s+', ' ', text)  
text = text.lower() # Now apply lower() to the string
#text = re.sub(r'\s+', ' ', text)
#text = re.sub(r'\([^)]*\)', '', text)
#text = re.sub(r'/',' ',text)
#text = re.sub("(\d+)","",text) 
text = re.sub(r'\([^()]*\d+[^()]*\)', '', text)
text = re.sub(r'\[[^\[\]]*\d+[^\[\]]*\]', '', text)
text = text.replace("ark.", '')

stopwords=["Tablo","tablo","ark.","ark","ya","/","(1)"]
#print(stopwords)

text = re.sub(r'\b(' + '|'.join(stopwords) + r')\b', '', text)

    
def text_summarizer(text, ratio=0.4):

    sent_list = sent_tokenize(text)
    word_list = " ".join(sent_list).split()
    word_freq = dict(nltk.FreqDist(word_list))
    G = nx.Graph()
    for word in word_freq:
        G.add_node(word, weight=word_freq[word])
    rank = nx.pagerank(G, alpha=0.85)    
    summarize_text = []
    for i in sorted(rank, key=rank.get, reverse=True):
        summarize_text.append(i)
    return " ".join(summarize_text[:int(len(summarize_text) * ratio)])

#TextRank
print(text_summarizer(text))
summary=text_summarizer(text)
# Assuming 'abstract' is defined somewhere earlier in your code
# from rouge import Rouge
# ROUGE = Rouge()

# print(ROUGE.get_scores(summary, abstract))

# print("orjinal özet: ", len(abstract))
# print("orjinal metin: ", len(text))
# print("olusturulan özet: ", len(summary))

# gercekmetinuzunlugu=len(text)
# özetuzunlugu=len(summary)
# orjinalozet=len(abstract)

import pandas as pd  # Import the Pandas library
# Duygu analizi için DataFrame oluşturma
sentences = sent_tokenize(summary)
manuel_df = pd.DataFrame(sentences, columns=['Metin'])
manuel_df['predicted_label'] = manuel_df['Metin'].apply(lambda x: classifier(x)[0]['label'])
manuel_df['confidence'] = manuel_df['Metin'].apply(lambda x: classifier(x)[0]['score'])

print("\nManuel yorumlar için duygu analizi sonuçları:")
print(manuel_df)
