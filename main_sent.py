from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
from prettytable import PrettyTable
# from model_distill_bert import getmodel
from utilities import compute_accuracy, compute_masks, mask_distillbert, get_model_distilbert, record_activations, mask_range_distilbert

batch_size = 256
mask_layer = 0
text_tag = "text"
compliment = True
results_table = PrettyTable()
if(compliment):
   results_table.field_names = results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "Base Complement Acc", "Base Compliment Conf", "STD Accuracy", "STD Confidence", "STD compliment ACC", "STD compliment Conf", "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf", "Total Masked", "Intersedction"]#, "Same as Max"]#"MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"
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

dataset_list = []
tokenizer = AutoTokenizer.from_pretrained("2O24dpower2024/distilbert-base-uncased-finetuned-emotion")
# Check if a GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the dataset
dataset_all = load_dataset("dair-ai/emotion")
avg_intersection = []
dataset_all1 = dataset_all['test']
record_dataset = dataset_all['train']
for j in range(0,6):
    # model = get_model_distilbert("esuriddick/distilbert-base-uncased-finetuned-emotion", mask_layer)
    
    model = get_model_distilbert("2O24dpower2024/distilbert-base-uncased-finetuned-emotion", mask_layer)
    dataset = dataset_all1.filter(lambda x: x['label'] in [j])
    dataset_complement = dataset_all1.filter(lambda x: x['label'] not in [j])
    dataset_record = record_dataset.filter(lambda x: x['label'] in [j])
    
    dataset2 = record_dataset.filter(lambda x: x['label'] not in [j])

    class_labels.append(f"Class {j}")
    acc = compute_accuracy(dataset, model, tokenizer, text_tag, batch_size=batch_size)
    dataset_list.append(acc[2])
    print("Class ",j, "base accuracy: ", acc[0], acc[1])
    base_accuracies.append(acc[0])
    base_confidences.append(acc[1])
    aug_dataset = acc[2]
    if(compliment):
        acc = compute_accuracy(dataset_complement, model, tokenizer, text_tag , batch_size=batch_size)
        print("Class ",j, "complement base accuracy: ", acc[0], acc[1])
        base_comp_acc.append(acc[0])
        base_comp_conf.append(acc[1])
        aug_dataset.extend(acc[2])
        
    print("Recording activations...")
    fc_vals = record_activations(dataset_record, model, tokenizer, text_tag=text_tag, mask_layer=mask_layer, batch_size=batch_size)
    fc_vals2 = record_activations(dataset2, model, tokenizer, text_tag=text_tag, mask_layer=mask_layer, batch_size=batch_size)

        
    mask_max, mask_std, mask_intersection, mask_max_low_std, mask_max_high_std, mask_std_high_max, mask_max_random_off = compute_masks(fc_vals,0.2)
    mask_std = mask_max_low_std
    print("Masking STD...")
    
    
    
    # model = mask_distillbert(model,mask_std)
    tao = 2.5
    model = mask_range_distilbert(tao,model, mask_max, fc_vals, fc_vals2)        
    
    t = int(mask_std.shape[0]-torch.count_nonzero(mask_std))
    print("Total Masked :", t)
    total_masked.append(t)
    diff_from_max.append(int((torch.logical_or(mask_std, mask_max) == 0).sum().item()))
    acc = compute_accuracy(dataset, model, tokenizer, text_tag, batch_size=batch_size, in_aug_dataset=aug_dataset[:len(dataset)]) 
    dataset_list.append(acc[2])
    print("accuracy after masking STD: ", acc[0], acc[1])
    std_accuracies.append(acc[0])
    std_confidences.append(acc[1])
    if(compliment):
        acc = compute_accuracy(dataset_complement, model, tokenizer, text_tag, batch_size=batch_size, in_aug_dataset=aug_dataset[len(dataset):])
        print("accuracy after masking STD on complement: ", acc[0], acc[1])
        std_comp_acc.append(acc[0])
        std_comp_conf.append(acc[1])
    model = get_model_distilbert("2O24dpower2024/distilbert-base-uncased-finetuned-emotion", mask_layer)
    tao = torch.inf
    print("Masking MAX...")
    # model = mask_distillbert(model,mask_max)
    model = mask_range_distilbert(tao,model, mask_max, fc_vals, fc_vals2)
    t = int(mask_max.shape[0]-torch.count_nonzero(mask_max))
    print("Total Masked :", t)
    acc = compute_accuracy(dataset, model, tokenizer, text_tag, batch_size=batch_size, in_aug_dataset=aug_dataset[:len(dataset)])
    dataset_list.append(acc[2])
    print("accuracy after masking MAX: ", acc[0], acc[1])
    max_accuracies.append(acc[0])
    max_confidences.append(acc[1])
    acc = compute_accuracy(dataset_complement, model, tokenizer, text_tag, batch_size=batch_size, in_aug_dataset=aug_dataset[len(dataset):])
    print("accuracy after masking MAX on complement: ", acc[0], acc[1])
    max_comp_acc.append(acc[0])
    max_comp_conf.append(acc[1])
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
print(results_table)
print("Layer ", mask_layer)
print("Average Base Accuracy: ",round(sum(base_accuracies)/len(base_accuracies), 4))
print("Average Base Confidence: ", round(sum(base_confidences)/len(base_confidences), 4))
print("Average STD Accuracy: ", round(sum(std_accuracies)/len(std_accuracies), 4))
print("Average STD Confidence: ", round(sum(std_confidences)/len(std_confidences), 4))
print("Average MAX Accuracy: ", round(sum(max_accuracies)/len(max_accuracies), 4))
print("Average MAX Confidence: ", round(sum(max_confidences)/len(max_confidences), 4))
print("Average STD Complement Accuracy: ", round(sum(std_comp_acc)/len(std_comp_acc), 4))
print("Average STD Complement Confidence: ", round(sum(std_comp_conf)/len(std_comp_conf), 4))
print("Average MAX Complement Accuracy: ", round(sum(max_comp_acc)/len(max_comp_acc), 4))
print("Average MAX Complement Confidence: ", round(sum(max_comp_conf)/len(max_comp_conf), 4))
print("Average Total Masked: ", round(sum(total_masked)/len(total_masked), 4))
print("Average Intersection: ", round(sum(diff_from_max)/len(diff_from_max), 4))
avg_intersection.append(round(sum(diff_from_max)/len(diff_from_max), 4))

print(avg_intersection)