from analytics_eda.core.numeric import select_normality_transforms

def test_select_candidate_transforms():
    # min > 0: include all
    stats = {'min': 1.0}
    alts = select_normality_transforms(stats)
    expected = {'yeo-johnson','arcsinh','box-cox','log','reciprocal','sqrt','log1p'}
    assert set(alts) == expected

    # min == 0: exclude box-cox/log/reciprocal
    stats = {'min': 0.0}
    alts = select_normality_transforms(stats)
    expected = {'yeo-johnson','arcsinh','sqrt','log1p'}
    assert set(alts) == expected

    # no min: only defaults
    alts = select_normality_transforms({})
    assert set(alts) == {'yeo-johnson','arcsinh'}
