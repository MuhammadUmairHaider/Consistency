import matplotlib.pyplot as plt
import numpy as np
import torch

from scipy import stats

def calculate_spearman_correlation(mask1, mask2):
    """
    Calculate Spearman correlation between two masks
    mask1, mask2: boolean tensors where 0 = masked, 1 = unmasked
    """
    # Convert to numpy arrays
    mask1_np = ~mask1.bool().cpu().numpy().astype(int)
    mask2_np = ~mask2.bool().cpu().numpy().astype(int)
    
    correlation, _ = stats.spearmanr(mask1_np, mask2_np)
    return correlation

def plot_neuron_correlations(all_fc_vals, num_classes):
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Generate x-axis values (percentages)
    percentages = np.array([0.05 * i for i in range(20)])
    
    # Store data for each class
    class_data = {}
    
    # Calculate correlations for each percentage and class
    for j in range(num_classes):
        spearman_corrs = []
        total_masked = []
        total_neurons = None  # Store the total number of neurons
        
        for per in percentages:
            # Compute masks
            mask_max, mask_std, mask_intersection, mask_max_low_std, \
            mask_max_high_std, mask_std_high_max, mask_max_random_off, \
            random_mask = compute_masks(all_fc_vals[j], per)
            
            if total_neurons is None:
                total_neurons = mask_std.numel()
            
            # Calculate Spearman correlation
            spearman_corr = calculate_spearman_correlation(mask_max, mask_max_low_std)
            # Count masked neurons (where mask value is 0)
            num_total_masked = (mask_std == 0).sum().item()
            
            spearman_corrs.append(spearman_corr)
            total_masked.append(num_total_masked)
        
        class_data[j] = {
            'spearman_correlation': spearman_corrs,
            'total_masked': total_masked,
            'total_neurons': total_neurons
        }
        
        # Plot lines for this class
        plt.plot(percentages, spearman_corrs, marker='o', 
                label=f'Spearman Correlation - Class {j}', linestyle='-')
        plt.plot(percentages, np.array(total_masked)/total_neurons, marker='s', 
                label=f'Masked Ratio - Class {j}', linestyle='--')
    
    # Customize the plot
    plt.xlabel('Percentage of Neurons')
    plt.ylabel('Spearman Correlation / Masked Ratio')
    plt.title('Spearman Correlation between Max and Std Masks vs Percentage')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Show plot
    plt.show()
    
    return class_data

def plot_individual_class_graphs(class_data, num_classes):
    # Calculate number of rows and columns for subplots
    num_rows = (num_classes + 2) // 3  # Ceiling division
    num_cols = min(3, num_classes)
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    if num_rows == 1:
        axes = [axes]  # Make axes indexable if only one row
    
    percentages = np.array([0.05 * i for i in range(20)])
    
    # Create individual plots for each class
    for j in range(num_classes):
        row = j // 3
        col = j % 3
        
        # Get total number of neurons from stored data
        # total_neurons = class_data[j]['total_neurons']
        
        # Plot data
        axes[row][col].plot(percentages, class_data[j]['spearman_correlation'], 
                           marker='o', label='Spearman Correlation', linestyle='-')
        
        # Plot masked ratio
        # masked_ratio = np.array(class_data[j]['total_masked'])/total_neurons
        # axes[row][col].plot(percentages, masked_ratio, 
        #                    marker='s', label='Masked Ratio', linestyle='--')
        
        axes[row][col].set_xlabel('Percentage of Neurons')
        axes[row][col].set_ylabel('Correlation')
        axes[row][col].set_title(f'Class {j}')
        axes[row][col].grid(True, linestyle='--', alpha=0.7)
        axes[row][col].legend()
        
        # Set y-axis limits for correlation plot
        axes[row][col].set_ylim([-0.5, 1.1])
    
    # Remove empty subplots if any
    for j in range(num_classes, num_rows * num_cols):
        row = j // 3
        col = j % 3
        fig.delaxes(axes[row][col])
    
    plt.tight_layout()
    plt.show()

def print_correlation_statistics(class_data, num_classes):
    print("\nDetailed Statistics:")
    percentages = [0.05 * i for i in range(1,20)]
    
    for j in range(num_classes):
        corrs = class_data[j]['spearman_correlation']
        masked_ratio = np.array(class_data[j]['total_masked'])/class_data[j]['total_neurons']
        
        print(f"\nClass {j}:")
        print(f"Max Correlation: {max(corrs):.3f} at {percentages[np.argmax(corrs)]:.2f}")
        print(f"Min Correlation: {min(corrs):.3f} at {percentages[np.argmin(corrs)]:.2f}")
        print(f"Mean Correlation: {np.mean(corrs):.3f}")
        print(f"Std Correlation: {np.std(corrs):.3f}")
        print(f"Final Masked Ratio: {masked_ratio[-1]:.3f}")

def create_correlation_plots(all_fc_vals, num_classes):
    print("Creating combined plot...")
    class_data = plot_neuron_correlations(all_fc_vals, num_classes)
    
    print("\nCreating individual class plots...")
    plot_individual_class_graphs(class_data, num_classes)
    
    print_correlation_statistics(class_data, num_classes)
    
    return class_data