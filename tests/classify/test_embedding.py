import pandas as pd
import numpy as np

from proxigenomics_toolkit.classify.embedding import center_of_mass

# Test functions are now standalone, no class needed for basic tests
def test_center_of_mass_single_embedding_correctness():
    # Create a sample DataFrame with one row and known values
    embedding_values = np.arange(0, 7.68, 0.01) # This will create 768 values
    data = {'col' + str(i): [embedding_values[i]] for i in range(768)}
    data['length'] = [150]
    embeds_single = pd.DataFrame(data)

    expected_result = embedding_values
    actual_result = center_of_mass(embeds_single)

    assert actual_result.shape == expected_result.shape, "Shape mismatch for single embedding"
    np.testing.assert_array_almost_equal(actual_result, expected_result, decimal=5,
                                         err_msg="Single embedding CoM calculation is incorrect")

def test_center_of_mass_multiple_embeddings_correctness():
    embed_dim = 768
    data_multi = {
        **{'col' + str(i): [0.1, 0.2, 0.3] for i in range(embed_dim)},
        'length': [10, 20, 30]
    }
    embeds_multiple = pd.DataFrame(data_multi)

    expected_value_per_dimension = ( (0.1 * 10) + (0.2 * 20) + (0.3 * 30) ) / (10 + 20 + 30)
    expected_multiple = np.full(embed_dim, expected_value_per_dimension)

    actual_multiple = center_of_mass(embeds_multiple)
    
    assert actual_multiple.shape == expected_multiple.shape, "Shape mismatch for multiple embeddings"
    np.testing.assert_array_almost_equal(actual_multiple, expected_multiple, decimal=5,
                                         err_msg="Multiple embeddings CoM calculation is incorrect")

def test_center_of_mass_zero_length_contribution():
    embed_dim = 768
    data = {
         **{'col' + str(i): [0.1, 0.5, 0.3] for i in range(embed_dim)},
        'length': [10, 0, 20]
    }
    embeds_zero_length = pd.DataFrame(data)

    expected_value = ((0.1 * 10) + (0.3 * 20)) / (10 + 20)
    expected_result = np.full(embed_dim, expected_value)

    actual_result = center_of_mass(embeds_zero_length)
    
    assert actual_result.shape == expected_result.shape, "Shape mismatch for zero length test"
    np.testing.assert_array_almost_equal(actual_result, expected_result, decimal=5,
                                         err_msg="CoM calculation with zero length contribution is incorrect")
