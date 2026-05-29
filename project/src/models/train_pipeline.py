from src.data.download import download_data
from src.data.process import process_data
from src.models.learn import train_model


def main():
    download_data()
    process_data()
    train_model()


if __name__ == "__main__":
    main()