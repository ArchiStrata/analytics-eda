from analytics_eda.core.categorical.categorical_inferential_analysis import categorical_inferential_analysis

def make_freq_table(freq_counts):
    """
    Convert a flat count dict into the nested frequency_table format:
        { category: {'count': int, 'proportion': float}, ... }
    Returns (frequency_table, total_count)
    """
    total = sum(freq_counts.values())
    freq_table = {
        cat: {'count': count, 'proportion': count / total}
        for cat, count in freq_counts.items()
    }
    return freq_table, total

def test_uniform_distribution_does_not_reject_null():
    counts = {'A': 25, 'B': 25, 'C': 25, 'D': 25}
    freq_table, total = make_freq_table(counts)
    result = categorical_inferential_analysis(freq_table, total)
    gof = result['goodness_of_fit']

    assert gof['p_value'] > gof['alpha']
    assert gof['reject_null_uniform'] is False

def test_skewed_distribution_rejects_null():
    counts = {'A': 70, 'B': 10, 'C': 10, 'D': 10}
    freq_table, total = make_freq_table(counts)
    result = categorical_inferential_analysis(freq_table, total)
    gof = result['goodness_of_fit']

    assert gof['p_value'] < gof['alpha']
    assert gof['reject_null_uniform'] is True

def test_chi2_stat_and_pvalue_types():
    counts = {'X': 60, 'Y': 40}
    freq_table, total = make_freq_table(counts)
    result = categorical_inferential_analysis(freq_table, total)
    gof = result['goodness_of_fit']

    assert isinstance(gof['chi2_statistic'], float)
    assert isinstance(gof['p_value'], float)
    assert isinstance(gof['reject_null_uniform'], bool)

def test_custom_alpha_threshold():
    counts = {'cat': 10, 'dog': 80, 'bird': 10}
    freq_table, total = make_freq_table(counts)
    alpha = 0.10
    result = categorical_inferential_analysis(freq_table, total, alpha=alpha)
    gof = result['goodness_of_fit']

    assert gof['alpha'] == alpha
