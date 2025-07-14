from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Assuming the functions are in a file named labeller.py
from proxigenomics_toolkit.classify.labeller import (
    anti_join,
    exclude_clusters,
    high_quality_clusters,
    identify_suspected_intra,
    normalised_out_degree,
    replace_zeros,
    scaler,
    seq2cluster_similarity,
    transform,
)


def test_scaler_no_mu_sig():
    """Test scaler when mu and sigma are not provided."""
    arr = np.array([1, 2, 3, 4, 5])
    scaled_arr, mu, sig = scaler(arr)
    assert np.isclose(mu, 3.0)
    assert np.isclose(sig, np.std(arr))
    assert np.allclose(scaled_arr, (arr - 3.0) / np.std(arr))
    assert np.isclose(scaled_arr.mean(), 0)
    assert np.isclose(scaled_arr.std(), 1)


def test_scaler_with_mu_sig():
    """Test scaler when mu and sigma are provided."""
    arr = np.array([1, 2, 3, 4, 5])
    mu, sig = 2.5, 1.5
    scaled_arr = scaler(arr, mu, sig)
    assert np.allclose(scaled_arr, (arr - mu) / sig)


def test_transform():
    """Test the transform function."""
    data = {
        'intra': [0, 1, 0],
        'cov_u': [10, 20, 30],
        'cov_v': [1, 2, 3],
        'contacts': [100, 200, 300],
        'sites_u': [1000, 2000, 1500],
        'sites_v': [1200, 2200, 1800],
        'uf_u': [0.8, 0.9, 0.85],
        'uf_v': [0.9, 0.95, 0.88]
    }
    df = pd.DataFrame(data)
    transformed_df = transform(df)
    assert 'intra_z' in transformed_df.columns
    assert 'cov_z' in transformed_df.columns
    assert 'freq_z' in transformed_df.columns
    assert transformed_df['intra_z'].dtype == np.uint8
    assert np.all(transformed_df['intra_z'] == df['intra'])
    assert np.isclose(transformed_df['cov_z'].values.mean(), 0)
    assert np.isclose(transformed_df['cov_z'].values.std(), 1, atol=1e-5)
    assert np.isclose(transformed_df['freq_z'].values.mean(), 0)
    assert np.isclose(transformed_df['freq_z'].values.std(), 1, atol=1e-5)


def test_anti_join():
    """Test the anti_join function."""
    target_data = {'seq': ['A', 'B', 'C', 'D'], 'cluster': [1, 2, 1, 2], 'value': [10, 20, 30, 40]}
    target_df = pd.DataFrame(target_data)
    exclude_data = {'seq': ['B', 'D', 'E'], 'cluster': [2, 2, 3]}
    exclude_df = pd.DataFrame(exclude_data)

    result_df = anti_join(target_df, exclude_df)
    expected_data = {'seq': ['A', 'C'], 'cluster': [1, 1], 'value': [10, 30]}
    expected_df = pd.DataFrame(expected_data).set_index(['seq', 'cluster'])
    pd.testing.assert_frame_equal(result_df.set_index(['seq', 'cluster']), expected_df)


@patch('proxigenomics_toolkit.classify.labeller.pd.read_csv')
def test_high_quality_clusters(mock_read_csv):
    """Test the high_quality_clusters function."""
    header = pd.MultiIndex.from_product([['CheckMv1', 'CheckMv2'], ['Completeness', 'Contamination']])
    data = {
        ('CheckMv1', 'Completeness'): [95, 80, 99], ('CheckMv1', 'Contamination'): [5, 15, 2],
        ('CheckMv2', 'Completeness'): [96, 82, 98], ('CheckMv2', 'Contamination'): [4, 16, 3],
    }
    df = pd.DataFrame(data, index=['CL1', 'CL2', 'CL3'], columns=header)
    df.sort_index(axis=1, inplace=True)
    mock_read_csv.return_value = df
    result = high_quality_clusters('dummy.csv', 90, 10, 'CheckMv1')
    assert result == {'CL1', 'CL3'}


def test_exclude_clusters():
    """Test the exclude_clusters function."""
    data = {'cluster_name': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]}
    df = pd.DataFrame(data)
    accepted = {'A', 'C'}
    with patch('proxigenomics_toolkit.classify.labeller.logger'):
        result_df = exclude_clusters(df, accepted)
    expected_data = {'cluster_name': ['A', 'C'], 'value': [1, 3]}
    expected_df = pd.DataFrame(expected_data)
    pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df.reset_index(drop=True))


def test_identify_suspected_intra():
    """Test the identify_suspected_intra function."""
    data = {
        'seq': ['s1', 's1', 's2', 's2', 's3', 's3'],
        'cluster': ['c1', 'c2', 'c1', 'c2', 'c1', 'c3'],
        'cluster_name': ['c1', 'c2', 'c1', 'c2', 'c1', 'c3'],
        'contacts': [11, 5, 20, 2, 15, 30],
        'similarity': [0.91, 0.8, 0.95, 0.5, 0.92, 0.98],
        'length_u': [1000, 1000, 1200, 1200, 800, 800],
        'length_v': [5000, 6000, 5000, 6000, 5000, 7000]
    }
    df = pd.DataFrame(data)
    accepted_clusters = {'c1', 'c2', 'c3'}

    with patch('proxigenomics_toolkit.classify.labeller.logger'):
        result_df = identify_suspected_intra(
            df=df, accepted_clusters=accepted_clusters, min_similarity=0.9, min_contacts=10,
            min_cluster_length=4000, max_seq_length=1300, min_degree=2
        )

    expected_data = {
        'cluster_name': ['c3'], 'contacts': [30], 'similarity': [0.98],
        'length_u': [800], 'length_v': [7000]
    }
    expected_index = pd.MultiIndex.from_tuples([('s3', 'c3')], names=['seq', 'cluster'])
    expected_df = pd.DataFrame(expected_data, index=expected_index)
    pd.testing.assert_frame_equal(result_df[expected_df.columns], expected_df)


def test_seq2cluster_similarity():
    """Test the seq2cluster_similarity function."""
    from sklearn.preprocessing import normalize
    embeddings = MagicMock()
    seq_embeds = normalize(np.array([[0.1, 0.2], [0.3, 0.4]]))
    cluster_embeds = normalize(np.array([[0.5, 0.6], [0.7, 0.8]]))
    embeddings.seq_embeds = pd.DataFrame(seq_embeds, index=['seq1', 'seq2'])
    embeddings.cluster_embeds = pd.DataFrame(cluster_embeds, index=['CL1', 'CL2'])

    df = pd.DataFrame({'seq': ['seq1', 'seq2'], 'cluster': ['CL1', 'CL2']})

    with patch('proxigenomics_toolkit.classify.labeller.linear_kernel', new=lambda X, Y: np.dot(X, Y.T)):
        result = seq2cluster_similarity(df, embeddings, dimension=2)

    expected = np.array([np.dot(seq_embeds[0], cluster_embeds[0]),
                         np.dot(seq_embeds[1], cluster_embeds[1])])
    assert np.allclose(result, expected)


def test_normalised_out_degree():
    """Test the normalised_out_degree function."""
    data = {'contacts': [10, 20], 'sites_v': [100, 200], 'cov_v': [1.0, 1.5], 'uf_v': [0.9, 0.8]}
    x = pd.DataFrame(data)
    edge_weight = x.contacts / np.sqrt(x.sites_v * x.cov_v * x.uf_v)
    expected_result = edge_weight / edge_weight.sum()
    result = normalised_out_degree(x)
    assert np.allclose(result, expected_result)


def test_replace_zeros():
    """Test the replace_zeros function."""
    s = pd.Series([1.0, 2.0, 0.0, 4.0, 0.0])
    expected_s = pd.Series([1.0, 2.0, 0.1, 4.0, 0.1])
    result_s = replace_zeros(s, 0.1)
    pd.testing.assert_series_equal(result_s, expected_s)


def test_replace_zeros_no_zeros():
    """Test replace_zeros with a series containing no zeros."""
    s = pd.Series([1.0, 2.0, 3.0])
    pd.testing.assert_series_equal(s, replace_zeros(s, 0.1))


def test_replace_zeros_all_zeros():
    """Test replace_zeros with a series containing only zeros."""
    s = pd.Series([0.0, 0.0, 0.0])
    assert np.all(s == replace_zeros(s, 0.1))
