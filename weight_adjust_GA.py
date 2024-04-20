import numpy as np

def refine_weight_dict_by_GA(weight_dict, candidates_id_list, site_before_results_dict, site_after_results_dict, step_size=0.1, fair_metric='loss'):
    if fair_metric == 'acc':
        signal = -1.0
    elif fair_metric == 'loss':
        signal = 1.0
    else:
        raise ValueError('fair_metric must be acc or loss')
    
    value_list = []
    for id in candidates_id_list:
        value_list.append((site_after_results_dict[id].cpu() - site_before_results_dict[id].cpu()).item())

    value_list = np.array(value_list)
    
    step_size = 1/len(weight_dict) * step_size 
    norm_gap_list = value_list / np.max(np.abs(value_list))
    
    for i in range(len(candidates_id_list)):
        weight_dict[candidates_id_list[i]] += signal * norm_gap_list[i] * step_size 

    weight_dict = weight_clip(weight_dict) 
    
    return weight_dict


def weight_clip(weight_dict):
    new_total_weight = 0.0
    for key_name in weight_dict.keys():
        weight_dict[key_name] = np.clip(weight_dict[key_name], 0.0, 1.0)
        new_total_weight += weight_dict[key_name]
    
    for key_name in weight_dict.keys():
        weight_dict[key_name] /= new_total_weight
    
    return weight_dict