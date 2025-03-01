import pathlib
import pickle
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50


class ImageSimilarityFinder:
    def __init__(self, model_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.features_dict = {}
        self.model_path = model_path

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image for ResNet50."""
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        return img_tensor

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from image using ResNet50."""
        img = self.load_and_preprocess_image(image_path)
        with torch.no_grad():
            features = self.model(img)
        features = features.cpu().numpy().flatten()
        return features

    def build_features_database(self, images_directory: str) -> None:
        """Build features database from images in directory."""
        for img_path in pathlib.Path(images_directory).glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    features = self.extract_features(str(img_path))
                    self.features_dict[str(img_path)] = features
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

    def find_similar_images(
        self, query_image_path: str, num_results: int = 5
    ) -> List[Tuple[str, float]]:
        """Find similar images to query image."""
        query_features = self.extract_features(query_image_path)

        similarities = []
        for path, features in self.features_dict.items():
            similarity = cosine_similarity(
                query_features.reshape(1, -1), features.reshape(1, -1)
            )[0][0]
            similarities.append((path, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:num_results]

    def save_features(self, save_path: str) -> None:
        """Save features dictionary to file."""
        with open(save_path, "wb") as f:
            pickle.dump(self.features_dict, f)

    def load_features(self, load_path: str) -> None:
        """Load features dictionary from file."""
        with open(load_path, "rb") as f:
            self.features_dict = pickle.load(f)
