from logger import Logger
from similarity_finder import ImageSimilarityFinder

logger = Logger("query_image", log_file="logs/query_image.log")


def main(file_name: str):
    finder = ImageSimilarityFinder()

    finder.load_features("features.pkl")

    query_image = file_name
    logger.info(f"Finding images similar to {query_image}...")

    similar_images = finder.find_similar_images(query_image, num_results=5)

    print("\nResults:")
    for path, similarity in similar_images:
        print(f"Similarity: {similarity:.4f} - {path}")


if __name__ == "__main__":
    main("test_images/images.jpeg")
