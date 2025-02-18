import torch
import math

# input:tensor
# output:int/float
def get_entropy_of_dataset(tensor:torch.Tensor):
    """Calculate the entropy of the entire dataset"""
    labels = tensor[:,-1]
    _, counts = torch.unique(labels, return_counts=True)
    probabilities = counts.float() / counts.sum()
    entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-9))
    return round(entropy.item(), 4)
    
# input:tensor,attribute number 
# output:int/float
def get_avg_info_of_attribute(tensor:torch.Tensor, attribute:int):
    """Return avg_info of the attribute provided as parameter"""
    unique_values, counts = torch.unique(tensor[:,attribute], return_counts=True)
    avg_info = 0.0
    for value, count in zip(unique_values, counts):
        subset_tensor = tensor[tensor[:,attribute] == value]
        subset_entropy = get_entropy_of_dataset(subset_tensor)
        avg_info += (count.item() / tensor.size(0)) * subset_entropy
    return round(avg_info, 4)

# input:tensor,attribute number
# output:int/float
def get_information_gain(tensor:torch.Tensor, attribute:int):
    """Return Information Gain of the attribute provided as parameter"""
    total_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)
    information_gain = total_entropy - avg_info
    return round(information_gain, 6)
    
# input: tensor
# output: ({dict},int)
def get_selected_attribute(tensor:torch.Tensor):
    """
    Return a tuple with the first element as a dictionary which has IG of all columns
    and the second element as an integer representing attribute number of selected attribute

    example : ({0: 0.123, 1: 0.768, 2: 1.23} , 2)
    """
    num_attributes = tensor.size(1) - 1
    info_gain_dict = {}
    for i in range(num_attributes):
        info_gain_dict[i] = get_information_gain(tensor, i)
    selected_attribute = max(info_gain_dict, key=info_gain_dict.get)
    return info_gain_dict, selected_attribute