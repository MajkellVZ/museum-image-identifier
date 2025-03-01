from logger import Logger
from similarity_finder import ImageSimilarityFinder

logger = Logger("feature_builder", log_file="logs/feature_builder.log")


def main():
    finder = ImageSimilarityFinder()

    logger.info("Building features database...")
    finder.build_features_database("images")

    finder.save_features("features.pkl")

    logger.info("Features built and stored")


if __name__ == "__main__":
    main()
