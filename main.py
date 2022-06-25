from typing import Text
from fastapi import FastAPI
from pydantic import BaseModel
from textblob import TextBlob
import preprocessor as p
import unidecode
from transformers import pipeline
from transformers import  BertTokenizer
from transformers import BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import uvicorn
import nltk


################# CONST VALUES & GLOBAL VARIABLES ############

BETO_PATH='beto'

app = FastAPI()

toxicity_scorer=pipeline("text-classification", model="Newtral/xlm-r-finetuned-toxic-political-tweets-es")

tokenizer = BertTokenizer.from_pretrained(BETO_PATH,do_lower_case=True)
model = BertForSequenceClassification.from_pretrained(
          BETO_PATH, 
          num_labels=2, 
          output_attentions=False,
          output_hidden_states=False
        )
model.load_state_dict(torch.load('torch-models/claim_detection.pth',map_location='cpu'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.eval()
TEST_MODEL=False



################# AUX FUNCTIONS #############################
def clean_tweet(text):
    clear_text = text.replace('\n', ' ').replace('\r', '').replace('\t', '')
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.ESCAPE_CHAR)
    clear_text = p.clean(unidecode.unidecode(clear_text.lower()))
    return clear_text

def tokenize (text):
    encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True, 
            max_length=280,
            truncation=True,
            padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt'
          )
    input_ids=encoding['input_ids'].to(device)
    attention_mask=encoding['attention_mask'].to(device)
    return (input_ids, attention_mask)

def predict(input_ids,att_mask):
    outputs = model(input_ids,att_mask)
    logits=outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=1).flatten()
    return result

#-------------TEST MODEL - TEST_MODEL=True--------------------
def create_data_loader(input_ids,att_masks,labels,batch_size,num_workers):
    data = TensorDataset(input_ids,att_masks,labels)
    sampler = SequentialSampler(data)
    dataloader = DataLoader(
        data, 
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers)
    return dataloader

def prepare_data_set(dataset):
    input_ids = []
    attention_mask = []
    for i in dataset.index:
      encoding = tokenizer.encode_plus(
          dataset['tweet'][i],
          add_special_tokens=True, 
          max_length=280,
          truncation=True,
          padding='max_length')
      input_ids.append(encoding['input_ids'])
      attention_mask.append(encoding['attention_mask'])
    return (torch.tensor(input_ids), torch.tensor(attention_mask))

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def clean_dataset_tweet_text(df,key):
  # Configuramos tweet-preprocessor para que nos elimine del tweet: Urls, menciones, emoticonos, etc.
  p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.HASHTAG, p.OPT.EMOJI, p.OPT.SMILEY, p.OPT.ESCAPE_CHAR)

  df2 = pd.DataFrame(df)
  # Limpiamos los tweets
  for i in tqdm(range(df2.shape[0])):
      # Limpiamos el tweet con tweet-preprocessor y lo pasamos a minúsculas
      df2.loc[i, 'clean_'+key] = (p.clean(unidecode.unidecode(df2.iloc[i][key].lower())))
  
  return df2

def test_model():
    test_data_url = (r'https://raw.githubusercontent.com/Newtral-Tech/clef2021-checkthat/main/data/csv_no_linebreak/dataset_dev.csv')
    df_test = pd.read_csv(test_data_url)
    df_test=clean_dataset_tweet_text(df_test,'tweet_text')
    df_test.drop(['topic_id','tweet_id','tweet_url','tweet_text'], inplace=True, axis=1)
    df_test.rename(columns = {'clean_tweet_text':'tweet'}, inplace = True)
    test_input_ids, test_att_mask = prepare_data_set(df_test)
    test_labels=torch.tensor(df_test['claim'])
    test_dataloader=create_data_loader(test_input_ids, test_att_mask,test_labels,32,2)

    model.eval()

    accuracy = 0
    all_logits = []
    all_labels = []
    for step, batch in enumerate(test_dataloader):
        batch_input_ids, batch_input_mask, batch_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            outputs = model(batch_input_ids,token_type_ids=None, attention_mask=batch_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        batch_labels = batch_labels.to('cpu').numpy()
        predict_labels = np.argmax(logits, axis=1).flatten()
        all_logits.extend(predict_labels.tolist())
        all_labels.extend(batch_labels.tolist())
        tmp_eval_accuracy = flat_accuracy(logits, batch_labels)
        accuracy += tmp_eval_accuracy
        print(step)
    print("  Accuracy: {0:.5f}".format(accuracy / (step+1)))
#-------------TEST MODEL -------------------------------------    

################# JSON MODELS ###############################

class TweetText(BaseModel):
    text: str

################# INTERNAL EVENTS FUNCTIONS #################

@app.on_event("startup")
async def startup_event():
    if TEST_MODEL:
        test_model()
    nltk.download('brown')
    nltk.download('punkt')

    
################# ENDPOINTS #################################
@app.get("/check-avaiability")
async def root():
    return {"message": "OK"}

    
@app.post("/analyze")
async def analyze(tweet : TweetText):
    #cleaning tweet text
    clean_text = clean_tweet(tweet.text)
    #polarity-subjetivity 
    translated_text=TextBlob(clean_text).translate(from_lang='es',to='en')
    polarity=round(translated_text.sentiment.polarity,2)
    subj=round(translated_text.sentiment.subjectivity,2)
    #toxicity
    toxicity_score = round(toxicity_scorer(clean_text,top_k=1)[0]['score'],2)
    #claim score
    input_ids,att_mask=tokenize(clean_text)
    claim = predict(input_ids,att_mask)[0].item()
    print(claim)

    return {
        "message": "OK",
        'text':clean_text,
        'polarity':polarity,
        'subj':subj,
        'toxicity_score':toxicity_score,
        'claim':claim
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
