from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import nltk

nltk.download('punkt')

app = FastAPI()

# Türkçe Varlık Tespiti (NER) için gerekli model ve tokenizer'ı yükle
tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-ner-cased")
model = AutoModelForTokenClassification.from_pretrained("savasy/bert-base-turkish-ner-cased")
ner = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)

# Sentiment Analysis için pipeline'ı başlat
sentiment_analyzer = pipeline("sentiment-analysis")

class TextRequest(BaseModel):
    text: str  # Bu modelin anahtarı "text" olarak güncellenmeli

def classify_sentiment(text):
    result = sentiment_analyzer(text)
    label = result[0]['label']

    # Türkçe duygu etiketlerini belirleyin
    if label.lower() == 'positive':
        label = 'olumlu'
    elif label.lower() == 'negative':
        label = 'olumsuz'
    else:
        label = 'nötr'

    return label

def extract_entities(text):
    entities = ner(text)
    entity_list = [entity['word'] for entity in entities]
    return entity_list

@app.post("/process_review/")
async def process_review(request: TextRequest):
    text = request.text  # "text" anahtarını kullanarak metni alın

    # Varlıkları çıkar
    entities = extract_entities(text)

    # Her varlık için duygu analizi yap
    results = []
    for entity in entities:
        entity_sentiment = classify_sentiment(entity)
        results.append({"entity": entity, "sentiment": entity_sentiment})

    # Çıktıyı hazırla
    output = {
        "entity_list": entities,
        "results": results
    }

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9524)
