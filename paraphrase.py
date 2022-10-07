#Source: https://huggingface.co/springml111/Pegasus_Paraphrase_model
import torch
from transformers import (PegasusForConditionalGeneration, PegasusTokenizer)


best_model_path = "springml111/Pegasus_Paraphrase_model"
model = PegasusForConditionalGeneration.from_pretrained(best_model_path)
tokenizer = PegasusTokenizer.from_pretrained("springml111/Pegasus_Paraphrase_model")

def tokenize_data(text):
    # Tokenize the review body
    input_ = str(text) + ' </s>'
    max_len = 64
    # tokenize inputs
    tokenized_inputs = tokenizer(input_, padding='max_length', truncation=True, max_length=max_len, return_attention_mask=True, return_tensors='pt')

    inputs={"input_ids": tokenized_inputs['input_ids'],
        "attention_mask": tokenized_inputs['attention_mask']}
    return inputs

def generate_answers(text):
    inputs = tokenize_data(text)
    results= model.generate(input_ids= inputs['input_ids'], attention_mask=inputs['attention_mask'], do_sample=True,
                            max_length=64,
                            top_k=120,
                            top_p=0.98,
                            early_stopping=True,
                            num_return_sequences=1)
    answer = tokenizer.decode(results[0], skip_special_tokens=True)
    return answer

body = '''
The college was established during the formation of Ontario's college system in 1967. Colleges of Applied Arts and Technology were established on May 21, 1965, when the Ontario system of public colleges was created. 
'''
result =generate_answers(body)
print (result)