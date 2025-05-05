import torch
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class LinearProbe(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearProbe, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        return self.linear(x)
# L1 and L2 penalty functions
def l1_penalty(weights):
    return torch.sum(torch.abs(weights))

def l2_penalty(weights):
    return torch.sum(weights**2)

def get_pared_dataset(dataset):
    all_fc_vals = dataset
    paired_data = []
    for idx, vector in enumerate(all_fc_vals):
    # For each vector in the input, create a tuple of (vector, target_index)
        for item in vector:
            paired_data.append((item, idx))

    # Now you can shuffle this list
    import random
    random.shuffle(paired_data)
    return paired_data

def train_probe(
# Hyperparameters
dataset,
lambda_l1 = 0,
lambda_l2 = 0,
num_epochs = 10,
batch_size = 32,
learning_rate = 0.001,
input_size = 768,
num_classes = 6):
    use_gpu = torch.cuda.is_available()
    
    num_classes = len(dataset)


    dataset = get_pared_dataset(dataset)
    # Convert paired data to tensors
    inputs, targets = zip(*dataset)  # Unzip the paired data we created earlier
    inputs = torch.tensor(inputs).float()
    targets = torch.tensor(targets).long()

    # Create dataset and dataloader
    dataset = TensorDataset(inputs, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and criterion
    probe = LinearProbe(input_size, num_classes)
    if use_gpu:
        probe = probe.cuda()

    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'max', patience=1, factor=0.7)

    # Training loop
    best_accuracy = 0
    torch.save(probe.state_dict(), 'best_model.pth')

    for epoch in range(num_epochs):
        # Training Phase
        probe.train()
        avg_loss = 0
        num_tokens = 0
        
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]"):
            if use_gpu:
                batch_inputs = batch_inputs.cuda()
                batch_labels = batch_labels.cuda()
                
            
            num_tokens += batch_inputs.shape[0]
            
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = probe(batch_inputs)
            
            weights = list(probe.parameters())[0]
            loss = (
                criterion(outputs, batch_labels)
                + lambda_l1 * l1_penalty(weights)
                + lambda_l2 * l2_penalty(weights)
            )
            
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
        
        # Print epoch statistics
        print(f"Epoch: [{epoch + 1}/{num_epochs}], Loss: {avg_loss / num_tokens:.4f}")
        
        # Evaluation Phase
        probe.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_inputs, batch_labels in train_loader:
                if use_gpu:
                    batch_inputs = batch_inputs.cuda()
                    batch_labels = batch_labels.cuda()
                    
                outputs = probe(batch_inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Accuracy at epoch [{epoch + 1}/{num_epochs}]: {accuracy:.2f}%')
        
        # Update scheduler and save best model
        scheduler.step(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(probe.state_dict(), 'best_model.pth')
            
        # Early stopping conditions
        if best_accuracy > 80 and accuracy < 80:
            probe.load_state_dict(torch.load('best_model.pth'))
        if accuracy == 100.0:
            break

    print(f"Best accuracy achieved: {best_accuracy:.2f}%")
    
    #return best model
    probe.load_state_dict(torch.load('best_model.pth'))
    
    return probe