import pytest

from cminbpe.base import get_stats, merge, render_token, replace_control_characters


@pytest.fixture(scope="module")
def token_ids():
    return [1, 2, 3, 2, 3]


@pytest.fixture(scope="module")
def input_str_with_control_characters():
    # return along with the expected output string for testing
    return (
        "Hello\x00World\x01Test\x02String",
        "Hello\\u0000World\\u0001Test\\u0002String",
    )


# Test get_stats function
# start from empty counts
def test_get_stats_empty_counts(token_ids):
    expected_counts = {(1, 2): 1, (2, 3): 2, (3, 2): 1}
    counts = get_stats(token_ids, {})
    assert counts == expected_counts


# start from existing counts
def test_get_stats_existing_counts(token_ids):
    existing_counts = {(1, 2): 1}
    expected_counts = {(1, 2): 2, (2, 3): 2, (3, 2): 1}
    counts = get_stats(token_ids, existing_counts)
    assert counts == expected_counts


# Test merge function
# test with matching pair
def test_merge_matching_pair(token_ids):
    pair = (2, 3)
    idx = 4
    expected_ids = [1, 4, 4]
    new_ids = merge(token_ids, pair, idx)
    assert new_ids == expected_ids


# test with non-matching pair
def test_merge_non_matching_pair(token_ids):
    pair = (3, 4)
    idx = 5
    expected_ids = token_ids  # no change expected
    new_ids = merge(token_ids, pair, idx)
    assert new_ids == expected_ids


# Test with multiple control characters
def test_replace_multiple_control_characters(input_str_with_control_characters):
    input_str, expected_output = input_str_with_control_characters
    output_str = replace_control_characters(input_str)
    assert output_str == expected_output


# Test with no control characters
def test_replace_no_control_characters():
    input_str = "Hello World! This is a test string."
    expected_output = input_str  # no change expected
    output_str = replace_control_characters(input_str)
    assert output_str == expected_output


# Test render token function
def test_render_token():
    # Test with a regular string
    token = bytes("Hello", "utf-8")
    expected_output = "Hello"
    assert render_token(token) == expected_output
