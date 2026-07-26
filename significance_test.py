# significance_test.py
import numpy as np
from scipy import stats


def significance_test(paired_data: dict):
    """
    paired_data: {'method_a': [values], 'method_b': [values]}
    """
    method_a_values = paired_data['method_a']
    method_b_values = paired_data['method_b']
    
    t_stat, p_value = stats.ttest_rel(method_b_values, method_a_values)
    
    w_stat, p_value_wilcoxon = stats.wilcoxon(method_b_values, method_a_values)
    
    return {
        't_stat': t_stat,
        'p_value': p_value,
        'p_value_wilcoxon': p_value_wilcoxon,
        'significant': p_value < 0.05,
        'improvement': (np.mean(method_b_values) - np.mean(method_a_values)) / np.mean(method_a_values) * 100
    }