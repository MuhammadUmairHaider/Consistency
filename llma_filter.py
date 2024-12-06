from huggingface_hub import login
login("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")


from models.lama import LlamaForCausalLM
import torch


from transformers import AutoTokenizer, AutoModelForCausalLM

model1 = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

model1.config.m_layer = 27
import os

base_path = os.path.join("model_weights", 'llama-emotion-classification')
if not os.path.exists(base_path):
    os.makedirs(base_path)

weights_path = os.path.join(base_path, "weights.pth")

# torch.save(model1.state_dict(), weights_path)

model = LlamaForCausalLM(model1.config)

model.load_state_dict(torch.load(weights_path))


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

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
label_mapping = {0: 'neg', 1: 'pos'}

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
class_counts = {0: 0, 1: 0}

for example in progress_bar:
    prompt = '''Choose from one of these categories: neg, pos. Be careful distinguishing between similar categories.
{{The only reason I DVRd this movie was because 1. I live in Cleveland and Shaq plays basketball for us now and 2. I've always heard how awful it was. The movie did not disappoint. The best parts were Shaq's outfits. The worst parts were, well, just about everything else. My 12 year old son and I just squirmed and couldn't look at the screen when Shaq started rapping and we kept wondering why Max didn't wish for Kazzam to fix that front tooth of his! But for all it's terribleness we just couldn't stop watching it, the story sucked you in, like a black hole or quicksand or a tar pit, it was hypnotic. But it was worth it for the laughs and just to say that we actually watched "Kazzam".:neg}}
{{Timberlake's performance almost made attack the screen. It wasn't all bad, I just think the reporters role was wrong for him.<br /><br />LL Cool J played the typical rapper role, toughest,baddest guy around. I don't think the cracked a smile in the whole movie, not even when proposed to his girlfriend.<br /><br />Morgan Freeman pretty much carried the whole movie. He was has some funny scenes which are the high point of the movie.<br /><br />Kevin Spacey wasn't good or bad he was just "there".<br /><br />Overall it's a Dull movie. bad plot. a lot of bad acting or wrong roles for actors.:neg}}
{{I loved this series when it was on Kids WB, I didn't believe that there was a Batman spin off seeing as the original show ended in 1995 and this show came in 1997. First of all I loved the idea of Robin leaving Batman to solve crime on his own. It was an interesting perspective to their relationship. I also liked the addition of Tim Drake in the series, and once again like it's predecessor this show had great story lines, great animation (better then the original), fantastic voice work and of course brilliant writing. The only thing that I didn't like was that was when it was in the US it would often run episodes in a 15 minute storyline. I just wish some of the episodes could be longer. My favorite episode of any Batman cartoons comes in this series, and it's called "Over the Edge", in my opinion as good if not better then "Heart of Ice" and "Robin's reckoning." Overall a nice follow up, along with Superman this show made my childhood very happy.:pos}}
{{'Deliverance' is a brilliant condensed epic of a group of thoroughly modern men who embark on a canoe trip to briefly commune with nature, and instead have to fight for their sanity, their lives, and perhaps even their souls. The film has aged well. Despite being made in the early Seventies, it certainly doesn't look particularly dated. It still possesses a visceral punch and iconic status as a dramatic post-'Death of the Sixties' philosophical-and-cultural shock vehicle. There are very few films with similar conceits that can compare favourably to it, although the legendary Sam Peckinpah's stuff would have to be up there. Yes, there has been considerable debate and discussion about the film's most confronting scene (which I won't expand upon here) - and undoubtedly one of the most confronting scenes in the entire history of the cinematic medium - but what surprises about this film is how achingly beautiful it is at times. This seems to be generally overlooked (yet in retrospect quite understandably so). The cinematography that captures the essence of the vanishing, fragile river wilderness is often absolutely stunning, and it counterbalances the film as, in a moment of brief madness, we the viewers - along with the characters themselves - are plunged into unrelenting nightmare. 'Deliverance's narrative is fittingly lean and sinewy, and it is surprising how quickly events unfold from point of establishment, through to crisis, and aftermath. It all takes place very quickly, which lends a sense of very real urgency to the film. The setting is established effectively through the opening credits. The characters are all well-drawn despite limited time spent on back story. We know just enough about them to know them for the kind of man they are, like them and ultimately fear for them when all goes to hell. The conflict and violence within the movie seems to erupt out of nowhere, with a frightening lack of logic. This is author James Dickey's theme - that any prevailing romanticism about the nature of Man's perceived inherent 'goodness' can only wilt and die when his barely suppressed animal instincts come to the fore. There are no demons or bogeymen here. The predatory hillbillies - as the film's central villains - are merely crude, terrifyingly amoral cousins of our protagonists. They shock because their evil is petty and tangible. The film has no peripheral characters. All reflect something about the weaknesses and uncertainties of urbanised Homo Sapiens in the latter 20th century, and all are very real and recognisable. Burt Reynolds is wonderful in this movie as the gung-ho and almost fatally over-confident Survivalist, Lewis, and it is a shame to think that he really couldn't recapture his brief moment of dramatic glory throughout the rest of his still sputtering up-and-down career ('Boogie Nights' excluded, perhaps). Trust me, if your are not a Reynolds fan, you WILL be impressed with his performance here. John Voight is his usual effortlessly accomplished self, and Ned Beatty and Ronny Cox both make significant contributions. This is simply a great quartet of actors. To conclude, I must speculate as to if and when 'Deliverance' author James Dickey's 'To the White Sea' will be made. For those that enjoyed (?) this film, TTWS is a similarly harrowing tale of an American Air Force pilot's struggle for survival after being shot down over the Japanese mainland during WW2. It's more of the typically bleak existentialism and primordial savagery that is Dickey's trademark, but it has all the makings of a truly spectacular, poetic cinematic experience. There was the suggestion a few years ago that the Coen brothers might be producing it, but that eventually came to nothing. Being an avid Coen-o-phile it disappoints me to think what might have been had they gotten the green light on TTWS, rather than their last couple of relatively undistinguished efforts. Returning to 'Deliverance', it's impossible to imagine a movie of such honest, unnerving brutality being made in these times, and that is pretty shameful. We, the cinema-going public, are all the poorer for this.:pos}}
{{"{}":'''.format(example[text_tag])
    
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
with open('correct_predictions_IMDB.json', 'w') as f:
    json.dump(correct_predictions, f, indent=4)

print(f"Saved {len(correct_predictions)} correctly predicted examples to dataset")