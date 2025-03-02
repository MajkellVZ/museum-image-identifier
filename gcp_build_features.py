import os
import pickle

from logger import Logger
from similarity_finder import ImageSimilarityFinder

logger = Logger("gcp_feature_builder", log_file="logs/gcp_feature_builder.log")

bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET")
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")


def main():
    finder = ImageSimilarityFinder()

    logger.info("Building features database...")
    finder.build_features("images")

    logger.info("Uploading features to Cloud Storage...")

    data = finder.features_dict
    with open(f"/gcs/art-embeddings/data/embeddings.pkl", "wb") as f:
        pickle.dump(data, f)

    logger.info("Features built and stored in Cloud Storage")


if __name__ == "__main__":
    main()
