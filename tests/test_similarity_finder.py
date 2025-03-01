import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from similarity_finder import ImageSimilarityFinder


@pytest.fixture
def similarity_finder():
    return ImageSimilarityFinder()


@pytest.fixture
def test_images_dir():
    temp_dir = tempfile.mkdtemp()

    for i in range(3):
        img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
        img.save(Path(temp_dir) / f"test_image_{i}.jpg")

    yield temp_dir

    shutil.rmtree(temp_dir)


def test_init(similarity_finder):
    assert isinstance(similarity_finder.model, torch.nn.Sequential)
    assert similarity_finder.device in [torch.device("cuda"), torch.device("cpu")]
    assert isinstance(similarity_finder.features_dict, dict)


def test_load_and_preprocess_image(similarity_finder, test_images_dir):
    image_path = str(Path(test_images_dir) / "test_image_0.jpg")
    img_tensor = similarity_finder.load_and_preprocess_image(image_path)

    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.shape == (1, 3, 224, 224)


def test_extract_features(similarity_finder, test_images_dir):
    image_path = str(Path(test_images_dir) / "test_image_0.jpg")
    features = similarity_finder.extract_features(image_path)

    assert isinstance(features, np.ndarray)
    assert features.shape == (2048,)  # ResNet50 feature dimension


def test_build_features_database(similarity_finder, test_images_dir):
    similarity_finder.build_features(test_images_dir)

    assert len(similarity_finder.features_dict) == 3
    for path, features in similarity_finder.features_dict.items():
        assert isinstance(features, np.ndarray)
        assert features.shape == (2048,)


def test_find_similar_images(similarity_finder, test_images_dir):
    similarity_finder.build_features(test_images_dir)

    query_image = str(Path(test_images_dir) / "test_image_0.jpg")
    results = similarity_finder.find_similar_images(query_image, num_results=2)

    assert len(results) == 2
    assert isinstance(results[0], tuple)
    assert 0 <= results[0][1] <= 1


def test_save_and_load_features(similarity_finder, test_images_dir, tmp_path):
    similarity_finder.build_features(test_images_dir)
    features_file = tmp_path / "features.pkl"
    similarity_finder.save_features(str(features_file))

    new_finder = ImageSimilarityFinder()
    new_finder.load_features(str(features_file))

    assert similarity_finder.features_dict.keys() == new_finder.features_dict.keys()
    for key in similarity_finder.features_dict:
        np.testing.assert_array_equal(
            similarity_finder.features_dict[key], new_finder.features_dict[key]
        )


def test_invalid_image_path(similarity_finder):
    with pytest.raises(Exception):
        similarity_finder.load_and_preprocess_image("nonexistent.jpg")
