from huggingface_hub import login
login("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")


from models.gemma import GemmaForCausalLM
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM

model1 = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

model1.config.m_layer = 17
import os

base_path = os.path.join("model_weights", 'gemma')
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

# torch.save(model1.state_dict(), weights_path)

model = GemmaForCausalLM(model1.config)

model.load_state_dict(torch.load(weights_path))


tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")

import nethook
def manual_generate(model, input_ids, attention_mask, max_length):
    device = input_ids.device
    batch_size = input_ids.shape[0]
    
    # Initialize the output tensor with the input_ids
    generated = input_ids.clone()
    
    confidences = torch.tensor([])
    fc = torch.tensor([])
    with torch.no_grad():
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
                with nethook.TraceDict(model, ['model.mask_layer']) as ret:
                    outputs = model(input_ids=generated, attention_mask=attention_mask)
                    fc1_vals = [
                    ret[layer_fc1_vals].output[:,-1,:].to('cpu')
                    for layer_fc1_vals in ret
                ]
                    
                # all_fc_vals = torch.cat([all_fc_vals, torch.stack(fc1_vals, dim=1).unsqueeze(1)], dim=1)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply greedy decoding (argmax)
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append the confidence of the predicted token
                confidences = torch.cat([confidences, torch.nn.functional.softmax(next_token_logits, dim=-1).unsqueeze(1).to('cpu')], dim=1)
                
                # Append the new tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)], dim=1)
    
    return generated, confidences#, all_fc_vals

from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
import torch
import json
from collections import defaultdict
from queue import Queue
import numpy as np


from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm.auto import tqdm
import torch
import json

# Load dataset
dataset_all = load_dataset("stanfordnlp/imdb")['train']
# dataset_all = dataset_all.shuffle(seed=42).select(range(10000))

print(f"Loaded {len(dataset_all)} examples from SST2 dataset")
print(dataset_all.column_names)

# Create label mapping
label_mapping = {0: 'Company', 1: 'EducationalInstitution', 2: 'Artist', 3: 'Athlete', 4: 'OfficeHolder', 5: 'MeanOfTransportation', 6: 'Building', 7: 'NaturalPlace', 8: 'Village', 9: 'Animal', 10: 'Plant', 11: 'Album', 12: 'Film', 13: 'WrittenWork'}

# Initialize list to store correctly predicted examples
correct_predictions = []

# Combine train and test datasets
# train_data = dataset_all['train']
# test_data = dataset_all['test']
# combined_data = dataset_all['train']

# Configure progress bar for the combined dataset
progress_bar = tqdm(dataset_all, desc="Processing examples")

text_tag = 'text'
examples_per_class = 2000
i = 0
correct = 0
model.to('cuda')
model.eval()

# Make counter for each class
class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0}

for example in progress_bar:
    
    # example[text_tag] = '*'*400
    # only for IMDB
    # example[text_tag] = example[text_tag][:350]
    
    # Create prompt
    prompt = '''Choose from one of these categories: Company, EducationalInstitution, Artist, Athlete, OfficeHolder, MeanOfTransportation, Building, NaturalPlace, Village, Animal, Plant, Album, Film, WrittenWork. Be careful distinguishing between similar categories.

{{ Abbott of Farnham E D Abbott Limited was a British coachbuilding business based in Farnham Surrey trading under that name from 1929. A major part of their output was under sub-contract to motor vehicle manufacturers. Their business closed in 1972.:Company}}

{{ Dubai Gem Private School (DGPS) is a British school located in the Oud Metha area of Dubai United Arab Emirates. Dubai Gem Nursery is located in Jumeirah. Together the institutions enroll almost 1500 students aged 3 to 18.:EducationalInstitution}}

{{ Martin Marty McKinnon (born 5 July 1975 in Adelaide) is a former Australian rules footballer who played with Adelaide Geelong and the Brisbane Lions in the Australian Football League (AFL).McKinnon was recruited by Adelaide in the 1992 AFL Draft with their first ever national draft pick. He was the youngest player on Adelaide's list at the time and played for Central District in the SANFL when not appearing with Adelaide.:Athlete}}

{{ The Wedell-Williams XP-34 was a fighter aircraft design submitted to the United States Army Air Corps (USAAC) before World War II by Marguerite Clark Williams widow of millionaire Harry P. Williams former owner and co-founder of the Wedell-Williams Air Service Corporation.:MeanOfTransportation}}

{{"{}":'''.format(example['text'])
    
    input_ids = tokenizer([prompt, prompt], return_tensors='pt')
    
    if (class_counts[example['label']] >= examples_per_class):
        continue
    
    # Generate output
    output = manual_generate(
        model, 
        input_ids['input_ids'].to('cuda'), 
        input_ids['attention_mask'].to('cuda'), 
        input_ids['input_ids'].shape[1]+8
    )
    # print(example['text'])
    # print(tokenizer.decode(output[0][0][input_ids['input_ids'].shape[1]:]))
    # Get predicted label
    predicted_label = tokenizer.decode(
        output[0][0][input_ids['input_ids'].shape[1]:]
    ).split('}')[0]
    
    # strip perdicted label of "
    predicted_label = predicted_label.strip('"')
    
    # Get true label text using the mapping
    true_label_text = label_mapping[example['label']]
    
    i += 1
    if true_label_text == predicted_label:
        correct += 1
        class_counts[example['label']] += 1
        # Store correctly predicted example
        correct_predictions.append({
            'text': example[text_tag],
            'label': example['label']
        })
    else:
        print('Text:', example[text_tag])
        print('True label:', true_label_text)
        print('Predicted label:', predicted_label)
        print('-' * 50)
    
    # Update progress bar description with current accuracy
    progress_bar.set_description(f"Accuracy: {round(correct/i,3)*100}%")
    # if all class counts are equal to 1000, break
    if all(count == examples_per_class for count in class_counts.values()):
        break

print(f"Final accuracy: {round(correct/i,3)*100}%")

# Create new dataset from correct predictions
correct_dataset = Dataset.from_list(correct_predictions)

# Save the dataset
correct_dataset.save_to_disk('correct_predictions_dataset')

# Optionally, also save as JSON for easier inspection
with open('correct_gemma_predictions_db14.json', 'w') as f:
    json.dump(correct_predictions, f, indent=4)

print(f"Saved {len(correct_predictions)} correctly predicted examples to dataset")