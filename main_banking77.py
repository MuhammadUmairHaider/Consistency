from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import torch
from prettytable import PrettyTable
# from model_bert import getmodel
from utilities import compute_accuracy, compute_masks, mask_bert, get_model_bert

mask_layer = 11
text_tag = "text"
compliment = True
results_table = PrettyTable()
if(compliment):
   results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "Base Complement Acc", "Base Compliment Conf", "STD Accuracy", "STD Confidence", "STD compliment ACC", "STD compliment Conf", "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf", "Total Masked", "Intersedction"]#, "Same as Max"]#"MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"
# results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "STD Accuracy", "STD Confidence", "Same as Max"]#, "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"]

class_labels = []
base_accuracies = []
base_confidences = []
base_comp_acc = []
base_comp_conf = []
std_masked_counts = []
std_accuracies = []
std_confidences = []
std_comp_acc = []
std_comp_conf = []
max_masked_counts = []
max_accuracies = []
max_confidences = []
max_comp_acc = []
max_comp_conf = []
diff_from_max = []
total_masked = []


tokenizer = AutoTokenizer.from_pretrained("sharmax-vikas/bert-base-banking77-pt2")
# Check if a GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset_test = load_dataset("glue", "sst2")

# Load the dataset
dataset_all = load_dataset("PolyAI/banking77")
# Select the train split
dataset_all = dataset_all['train']

for j in range(0,77):
    model = get_model_bert("sharmax-vikas/bert-base-banking77-pt2", mask_layer)
    # model = AutoModelForSequenceClassification.from_pretrained("fabriceyhc/bert-base-uncased-dbpedia_14")
    dataset = dataset_all.filter(lambda x: x['label'] in [j])
    dataset_complement = dataset_all.filter(lambda x: x['label'] not in [j])
    
    if(j==76):
        dataset = dataset_all

    class_labels.append(f"Class {j}")
    acc = compute_accuracy(dataset, model, tokenizer, text_tag)
    print("Class ",j, "base accuracy: ", acc)
    base_accuracies.append(acc[0])
    base_confidences.append(acc[1])
    if(compliment):
        acc = compute_accuracy(dataset_complement, model, tokenizer, text_tag)
        print("Class ",j, "complement base accuracy: ", acc)
        base_comp_acc.append(acc[0])
        base_comp_conf.append(acc[1])
        

    #record the activations of the first fully connected layer, CLS tokken
    print("Recording activations...")
    progress_bar = tqdm(total=len(dataset))
    model.to(device)
    model.eval()
    fc_vals = []
    with torch.no_grad():
        for i in range(len(dataset)):
            text = dataset[i][text_tag]
            inputs = tokenizer(text, return_tensors="pt", max_length = 512).to(device)
            outputs = model(**inputs)
            fc_vals.append(outputs[1][mask_layer+1][:, 0].squeeze().cpu().numpy())
            progress_bar.update(1)
        progress_bar.close()


        
    mask_max, mask_std,mask_intersection, mask_max_low_std, mask_std_high_max = compute_masks(fc_vals,0.30)
    # mask_std = mask_max_low_std
    print("Masking STD...")
    model = mask_bert(model,mask_std)
    t = int(mask_std.shape[0]-torch.count_nonzero(mask_std))
    print("Total Masked :", t)
    total_masked.append(t)
    diff_from_max.append(115-(mask_std.shape[0]-torch.count_nonzero(mask_std)))
    acc = compute_accuracy(dataset, model, tokenizer, text_tag)
    print("accuracy after masking STD: ", acc)
    std_accuracies.append(acc[0])
    std_confidences.append(acc[1])
    if(compliment):
        acc = compute_accuracy(dataset_complement, model, tokenizer, text_tag)
        print("accuracy after masking STD on complement: ", acc)
        std_comp_acc.append(acc[0])
        std_comp_conf.append(acc[1])

    print("Masking MAX...")
    model = mask_bert(model,mask_max)
    t = int(mask_max.shape[0]-torch.count_nonzero(mask_max))
    print("Total Masked :", t)
    total_masked.append(t)
    acc = compute_accuracy(dataset, model, tokenizer, text_tag)
    print("accuracy after masking MAX: ", acc)
    max_accuracies.append(acc[0])
    max_confidences.append(acc[1])
    acc = compute_accuracy(dataset_complement, model, tokenizer, text_tag)
    print("accuracy after masking MAX on complement: ", acc)
    max_comp_acc.append(acc[0])
    max_comp_conf.append(acc[1])

    diff_from_max.append(int((torch.logical_or(mask_std, mask_max) == 0).sum().item()))
    if(compliment):
        results_table.add_row([
            class_labels[j],
            base_accuracies[j],
            base_confidences[j],
            base_comp_acc[j],
            base_comp_conf[j],
            std_accuracies[j],
            std_confidences[j],
            std_comp_acc[j],
            std_comp_conf[j],
            max_accuracies[j],
            max_confidences[j],
            max_comp_acc[j],
            max_comp_conf[j],
            total_masked[j],
            diff_from_max[j]
        ])
    # results_table.add_row([
    #     class_labels[j],
    #     base_accuracies[j],
    #     base_confidences[j],
    #     std_accuracies[j],
    #     std_confidences[j],
    #     # max_accuracies[j],
    #     # max_confidences[j],
    #     diff_from_max[j]
    # ])

print(results_table)