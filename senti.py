import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from lime.lime_text import LimeTextExplainer

tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
# classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer, framework='pt')

# explain the model decision
class_names = ['Very Positive', 'Positive', 'Neutral', 'Negative', 'Very Negative']
explainer = LimeTextExplainer(class_names=class_names)

reviews = pd.read_csv('./data/total_stars_text.csv')

text = reviews['text']
stars = reviews['stars']

output_file = open('./data/senti.csv', 'w')

# for i in range(len(reviews)):
#   t = text[i]
#   s = stars[i]

idx = random.randrange(0, len(reviews))


def predict(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    prob = torch.softmax(output['logits'], dim=1).detach()[0]
    prob = ','.join(['%.4f' % prob[idx] for idx in range(5)])
    return prob


review_to_pred = reviews.iloc(idx)
exp = explainer.explain_instance(review_to_pred, predict, num_features=20, num_samples=2000)
exp.show_in_notebook()
# out_str = '%d,%d,%s' % (i, s, prob)
# output_file.write(out_str + '\n')
