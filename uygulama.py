import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline

# 1. Veri kümesini hazırla
url = "https://raw.githubusercontent.com/anilkus/TRsemanticAnalysisTKNFST/main/e-ticaret_urun_yorumlari.csv"
df = pd.read_csv(url, sep=';', on_bad_lines='skip')

# Sütun isimlerini güncelle
df.rename(columns={'Metin': 'text', 'Durum': 'label'}, inplace=True)

# `2` olan etiketleri filtrele
df = df[df['label'] != 2]

# Etiketlerin doğru olduğundan emin olun
print("Etiketler:", df['label'].unique())
print(df.head())

# Veri kümesini `datasets` formatına dönüştür
dataset = Dataset.from_pandas(df)

# Dataset sütun isimlerini kontrol et
print("Dataset Sütunları:", dataset.column_names)

# 2. Modeli ve tokenizer'ı yükle
model_name = "savasy/bert-base-turkish-sentiment-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Veri kümesini yükleyin ve işle
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

# 4. Modeli fine-tuning yap
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,  # Eğitim epok sayısını 1'e düşürdük
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset,
    eval_dataset=encoded_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# Modeli kaydet
trainer.save_model("./fine-tuned-model")

# Fine-tuning işleminden sonra duygu analizi yapmak için pipeline oluştur
classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Yeni yorumlar üzerinde duygu analizi
new_reviews = [
    "Hızlı kargo, ürün kaliteli. Tavsiye ederim.",
    "Ürün beklediğim gibi çıkmadı, hayal kırıklığına uğradım.",
    "Fiyat/performans ürünü, memnunum.",
    "Kötü paketleme, ürün hasarlı geldi.",
    "Harika bir alışveriş deneyimiydi, teşekkürler."
]

# Her yorum için duygu analizi
for review in new_reviews:
    result = classifier(review)
    print(f"Review: {review}\nSentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}\n")
