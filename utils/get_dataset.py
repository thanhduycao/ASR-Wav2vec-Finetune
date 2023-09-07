from datasets import load_dataset
import csv
import zipfile
import gdown
import os
import argparse


def download_and_extract_data(url_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    gdown.download(url_path, output_path, quiet=False)
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(output_path.split(".")[0])


def generate_csv_from_dataset(zip_data, csv_file_path):
    # Write data to CSV using "|" as delimiter
    with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter="|")
        csv_writer.writerow(["path", "transcript"])  # Write header
        csv_writer.writerows(data)


def generate_zip_id_sentence(ds, id_name, sentence_name, data_path):
    ids = ds["train"][id_name]
    sentences = ds["train"][sentence_name]

    for i in range(len(ids)):
        ids[i] = data_path + ids[i] + ".wav"
    data = zip(ids, sentences)
    return data


def get_dataset(
    dataset_name,
    url_path,
    output_path,
    csv_file_path,
    id_name="id",
    sentence_name="sentence_norm",
):
    # Download and extract data
    download_and_extract_data(url_path, output_path)

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Generate CSV file
    csv_train_file_path = "train.csv"
    csv_train_file_path = os.path.join(csv_file_path, csv_train_file_path)
    csv_test_file_path = "test.csv"
    csv_test_file_path = os.path.join(csv_file_path, csv_test_file_path)

    train_zip_data = generate_zip_id_sentence(
        dataset, id_name, sentence_name, output_path
    )
    generate_csv_from_dataset(train_zip_data, csv_train_file_path)

    test_zip_data = generate_zip_id_sentence(
        dataset, id_name, sentence_name, output_path
    )
    generate_csv_from_dataset(test_zip_data, csv_test_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and extract data from a given url"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="quocanh34/soict_train_dataset",
        help="Name of the dataset to download",
    )
    parser.add_argument(
        "--url_path",
        type=str,
        default="https://drive.google.com/drive/folders/1FqCmmSjMMgkYjANXY7FD6tzqsfDwZJrY",
        help="URL path to download the data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="ASR-Wav2vec-Finetune/datasets",
        help="Path to store the downloaded data",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        default="ASR-Wav2vec-Finetune/datasets",
        help="Path to store the csv file",
    )
    parser.add_argument(
        "--id_name",
        type=str,
        default="id",
        help="Name of the column containing the id",
    )
    parser.add_argument(
        "--sentence_name",
        type=str,
        default="sentence_norm",
        help="Name of the column containing the sentence",
    )
    args = parser.parse_args()

    get_dataset(
        args.dataset_name,
        args.url_path,
        args.output_path,
        args.csv_file_path,
        args.id_name,
        args.sentence_name,
    )
