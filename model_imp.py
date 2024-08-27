@add_start_docstrings(
    """
    DistilBert Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    DISTILBERT_START_DOCSTRING,
)
class DistilBertForSequenceClassification(DistilBertPreTrainedModel):
    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        # Initialize weights and apply final processing
        self.post_init()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.distilbert.get_position_embeddings()

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if `new_num_position_embeddings != config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        self.distilbert.resize_position_embeddings(new_num_position_embeddings)

    @add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
import nethook
import transformers
from datasets import load_dataset
from tqdm import tqdm
import torch
from torch import nn
import numpy as np
from prettytable import PrettyTable
from typing import Optional, Union, Tuple


results_table = PrettyTable()
results_table.field_names = ["Class", "Base Accuracy", "Base Confidence", "Base Complement Acc", "Base Compliment Conf", "STD Accuracy", "STD Confidence", "STD compliment ACC", "STD compliment Conf", "MAX Accuracy", "MAX Confidence", "Max compliment acc", "Max compliment conf"]


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
class CustomModel(nn.Module):
    def __init__(self, model_name):
        super(CustomModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.hidden_size = self.model.config.hidden_size
        self.num_labels = self.model.config.num_labels
        
        # Create a masking layer
        self.masking_layer = nn.Parameter(torch.ones(self.hidden_size))
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[transformers.modeling_outputs.SequenceClassifierOutput, Tuple[torch.Tensor, ...]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        hidden_state = hidden_state * self.masking_layer
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return transformers.modeling_outputs.SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )
    
    
    
    
def getmodel():
    model = CustomModel("esuriddick/distilbert-base-uncased-finetuned-emotion")
    # model = AutoModelForSequenceClassification.from_pretrained("esuriddick/distilbert-base-uncased-finetuned-emotion")
    return model

tokenizer = AutoTokenizer.from_pretrained("esuriddick/distilbert-base-uncased-finetuned-emotion")
# Check if a GPU is available and use it
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_accuracy(dataset, model, tokenizer):
    correct = 0
    total_confidence = 0.0
    progress_bar = tqdm(total=len(dataset))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        for i in range(len(dataset)):
            text = dataset[i]['text']
            inputs = tokenizer(text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_idx = torch.argmax(predictions).item()
            if predicted_class_idx == dataset[i]['label']:
                correct += 1
                total_confidence += predictions[0][predicted_class_idx].item()
            progress_bar.update(1)
    progress_bar.close()
    
    accuracy = correct / len(dataset)
    average_confidence = total_confidence / correct if correct > 0 else 0.0
    
    return round(accuracy,4), round(average_confidence,4)

def apply_masks(model, masks):
    mask_indices = [mask.nonzero(as_tuple=True)[0] for mask in masks]

    # Mask the network
    i = 0
    for name, module in model.named_modules():
        if 'lin2' in name:
            for neuron_idx in mask_indices[i]:
                # Set the weights of the masked neurons to zero
                module.weight.data[neuron_idx] = torch.zeros_like(module.weight.data[neuron_idx])
            i += 1

    # Print a confirmation message
    print("Masking complete.")
    return model


# Load the dataset
dataset_all = load_dataset("dair-ai/emotion")
# Select the train split
dataset_all = dataset_all['train']
# Filter Classes
# class_label:
    # names:
    #     '0': sadness
    #     '1': joy
    #     '2': love
    #     '3': anger
    #     '4': fear
    #     '5': surprise
for j in range(0,6):
    model = getmodel()
    dataset = dataset_all.filter(lambda x: x['label'] in [j])
    dataset_complement = dataset_all.filter(lambda x: x['label'] not in [j])

    class_labels.append(f"Class {j}")
    acc = compute_accuracy(dataset, model, tokenizer)
    print("Class ",j, "base accuracy: ", acc)
    base_accuracies.append(acc[0])
    base_confidences.append(acc[1])
    acc = compute_accuracy(dataset_complement, model, tokenizer)
    print("Class ",j, "complement base accuracy: ", acc)
    base_comp_acc.append(acc[0])
    base_comp_conf.append(acc[1])
    

    #record the activations of the first fully connected layer, CLS tokken
    print("Recording activations...")
    l = []
    for n,m in model.named_modules():
        if("lin2" in n):
            l.append(n)
    correct = 0
    progress_bar = tqdm(total=len(dataset))
    model.to(device)
    model.eval()
    fc1_vals = []
    with torch.no_grad():
        for i in range(len(dataset)):
            text = dataset[i]['text']
            inputs = tokenizer(text, return_tensors="pt").to(device)
            with nethook.TraceDict(model, l) as ret:
                outputs = model(**inputs)
                fc1_vals_i = [
                    ret[layer_fc1_vals].output.mean(axis=1).squeeze().detach().cpu().numpy() #avg pooling
                    # ret[layer_fc1_vals].output[0][0].squeeze().detach().cpu().numpy() # CLS token
                    for layer_fc1_vals in ret]
                fc1_vals.append(fc1_vals_i)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_idx = torch.argmax(predictions).item()
                if(predicted_class_idx == dataset[i]['label']):
                    correct += 1
                progress_bar.update(1)
        progress_bar.close()
    # print("Class ",i, "base accuracy: ", correct / len(dataset))




    # Compute the mean and std of the activations
    fc1_vals_array = np.array(fc1_vals)
    mean_vals = np.mean(fc1_vals_array, axis=0)
    std_vals = np.std(fc1_vals_array, axis=0)

    mean_vals_tensor = torch.from_numpy(mean_vals)
    std_vals_tensor = torch.from_numpy(std_vals)


    masks_max = []
    for layer_std in mean_vals_tensor:
        # Sort the standard deviations and get the indices (ascending order)
        sorted_indices = torch.argsort(layer_std, descending=True)
        
        # Calculate the number of bottom 10% neurons
        bottom_10_percent_count = int(0.5 * layer_std.shape[0])
        
        # Create a mask with True for the bottom 10% neurons
        mask = torch.zeros_like(layer_std, dtype=torch.bool)
        mask[sorted_indices[:bottom_10_percent_count]] = True
        
        masks_max.append(mask)
        
    masks_std = []
    for layer_std in std_vals_tensor:
        # Sort the standard deviations and get the indices (ascending order)
        sorted_indices = torch.argsort(layer_std, descending=False)
        
        # Calculate the number of bottom 10% neurons
        bottom_10_percent_count = int(0.5 * layer_std.shape[0])
        
        # Create a mask with True for the bottom 10% neurons
        mask = torch.zeros_like(layer_std, dtype=torch.bool)
        mask[sorted_indices[:bottom_10_percent_count]] = True
        
        masks_std.append(mask)
        
    # mask std excluding the bottom 10% mean values

    # masks_std = []
    # for layer_std, layer_mean in zip(std_vals_tensor, mean_vals_tensor):
    #     # Sort the mean values and get the indices (ascending order)
    #     sorted_indices_mean = torch.argsort(layer_mean, descending=True)
        
    #     # Calculate the number of bottom 10% neurons
    #     bottom_10_percent_count = int(0.5 * layer_mean.shape[0])
        
    #     # Get the indices of the bottom 10% values in mean_vals_tensor
    #     bottom_10_percent_indices = sorted_indices_mean[:bottom_10_percent_count]
        
    #     # Create a mask to exclude the bottom 10% values from mean_vals_tensor
    #     exclusion_mask = torch.ones_like(layer_mean, dtype=torch.bool)
    #     exclusion_mask[bottom_10_percent_indices] = False
        
    #     # Sort the standard deviations and get the indices (ascending order)
    #     sorted_indices_std = torch.argsort(layer_std, descending=False)
        
    #     # Filter out the indices corresponding to the bottom 10% mean values
    #     filtered_indices_std = sorted_indices_std[exclusion_mask]
        
    #     # Calculate the number of bottom 10% neurons from the remaining neurons
    #     remaining_count = sorted_indices_std.shape[0]
    #     bottom_10_percent_remaining_count = int(0.2 * remaining_count)
        
    #     # Create a mask with True for the bottom 10% neurons from the filtered indices
    #     mask_std = torch.zeros_like(layer_std, dtype=torch.bool)
    #     mask_std[filtered_indices_std[:bottom_10_percent_remaining_count]] = True
        
    #     masks_std.append(mask_std)
        
    print("Masking STD...")
    model = apply_masks(model, masks_std)
    print("masked per layer: ", torch.count_nonzero(masks_std[0]))
    acc = compute_accuracy(dataset, model, tokenizer)
    print("accuracy after masking STD: ", acc)
    std_accuracies.append(acc[0])
    std_confidences.append(acc[1])
    acc = compute_accuracy(dataset_complement, model, tokenizer)
    print("accuracy after masking STD on complement: ", acc)
    std_comp_acc.append(acc[0])
    std_comp_conf.append(acc[1])
    
    model = getmodel()
    print("Masking MAX...")
    model = apply_masks(model, masks_max)
    print("masked per layer: ", torch.count_nonzero(masks_max[0]))
    acc = compute_accuracy(dataset, model, tokenizer)
    print("accuracy after masking MAX: ", acc)
    max_accuracies.append(acc[0])
    max_confidences.append(acc[1])
    acc = compute_accuracy(dataset_complement, model, tokenizer)
    print("accuracy after masking MAX on complement: ", acc)
    max_comp_acc.append(acc[0])
    max_comp_conf.append(acc[1])
    
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
        max_comp_conf[j]
    ])

print(results_table)

print(base_accuracies, "\n", base_confidences, "\n", base_comp_acc, "\n", base_comp_conf, "\n", std_accuracies, "\n", std_confidences, "\n", std_comp_acc, "\n", std_comp_conf, "\n", max_accuracies, "\n", max_confidences, "\n", max_comp_acc, "\n", max_comp_conf)