import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
  
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")


reviews = pd.read_csv('./data/total_stars_text.csv')

text = reviews['text']
stars = reviews['stars']

output_file = open('./data/senti.csv', 'w')

for i in range(len(reviews)):
  t = text[i]
  s = stars[i]

  try:
    encoded_input = tokenizer(t, return_tensors='pt')
    output = model(**encoded_input)
    output['logits']
  except Exception as e:
    print(e)
    continue

  prob = torch.softmax(output['logits'], dim=1).detach()[0]
  prob = ','.join(['%.4f' % prob[idx] for idx in range(5)])

  out_str = '%d,%d,%s' % (i, s, prob)
  output_file.write(out_str + '\n')

