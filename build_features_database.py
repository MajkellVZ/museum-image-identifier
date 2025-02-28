from similarity_finder import ImageSimilarityFinder


def main():
    finder = ImageSimilarityFinder()

    print("Building features database...")
    finder.build_features_database("images")

    finder.save_features("features.pkl")


if __name__ == "__main__":
    main()