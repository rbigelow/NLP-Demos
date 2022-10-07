#Source: https://medium.com/analytics-vidhya/text-summarization-using-bert-gpt2-xlnet-5ee80608e961

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from summarizer import Summarizer,TransformerSummarizer
body = '''
Georgian College is a College of Applied Arts and Technology in Ontario, Canada. Georigan has 13,000 full-time students, including 4,500 international students from 85 countries, across seven campuses, the largest being in Barrie.
The college was established during the formation of Ontario's college system in 1967. Colleges of Applied Arts and Technology were established on May 21, 1965, when the Ontario system of public colleges was created. 
Georgian College offers academic upgrading, apprenticeship training, certificate, diploma, graduate certificate, college degree and university programs (including combined degree-diplomas) and part-time studies in such areas such as automotive business, business and management, community safety, computer studies, design and visual arts, engineering technology and environmental studies, health, wellness and sciences, hospitality, tourism and recreation, human services, Indigenous studies, liberal arts, marine studies, and skilled trades.
'''
bert_model = Summarizer()
bert_summary = ''.join(bert_model(body, min_length=100, max_length=300))

model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
xlnet_summary = ''.join(model(body, min_length=100, max_length=300))

GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
gpt2_summary = ''.join(GPT2_model(body, min_length=100, max_length=300))

print("BERT----------------------------------------")
print(bert_summary)

print("XLNET----------------------------------------")
print(xlnet_summary)

print("GPT2----------------------------------------")
print(gpt2_summary)
print("--------------------------------------------")