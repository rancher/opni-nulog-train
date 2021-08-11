# Standard Library
import asyncio
import json
import logging
import os
import shutil

# Third Party
import boto3
import botocore
from botocore.client import Config
from botocore.exceptions import EndpointConnectionError
from NuLogParser import LogParser
from opni_nats import NatsWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")

S3_ENDPOINT = os.environ["S3_ENDPOINT"]
S3_ACCESS_KEY = os.environ["S3_ACCESS_KEY"]
S3_SECRET_KEY = os.environ["S3_SECRET_KEY"]
S3_BUCKET = os.getenv("S3_BUCKET", "opni_nulog_models")


async def train_nulog_model(s3_client, windows_folder_path):
    nr_epochs = 2
    num_samples = 0
    parser = LogParser()
    texts = parser.load_data(windows_folder_path)
    tokenized = parser.tokenize_data(texts, isTrain=True)
    parser.tokenizer.save_vocab()
    parser.train(tokenized, nr_epochs=nr_epochs, num_samples=num_samples)
    all_files = os.listdir("output/")
    if "nulog_model_latest.pt" in all_files and "vocab.txt" in all_files:
        logging.info("Completed training model")
        s3_client.meta.client.upload_file(
            "output/nulog_model_latest.pt", "nulog-models", "nulog_model_latest.pt"
        )
        s3_client.meta.client.upload_file(
            "output/vocab.txt", "nulog-models", "vocab.txt"
        )
        logging.info("Nulog model and vocab have been uploaded to Minio.")
    else:
        logging.info("Nulog model was not able to be trained and saved successfully.")
        return False
    return True


async def minio_setup_and_download_data(s3_client):
    try:
        s3_client.meta.client.download_file(
            "training-logs", "windows.tar.gz", "windows.tar.gz"
        )
        logging.info("Downloaded logs from s3 successfully")

        shutil.unpack_archive("windows.tar.gz", format="gztar")
    except EndpointConnectionError:
        logging.error(
            f"Could not connect to s3 with endpoint_url={S3_ENDPOINT}"
        )
        logging.info("Job failed")
        return False

    try:
        s3_client.meta.client.head_bucket(Bucket=S3_BUCKET)
        logging.info("nulog-models bucket exists")
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logging.info("{S3_BUCKET} bucket does not exist so creating it now")
            s3_client.create_bucket(Bucket=S3_BUCKET)
    return True


async def send_signal_to_nats():
    nw = NatsWrapper()
    nulog_payload = {
        "bucket": S3_BUCKET,
        "bucket_files": {
            "model_file": "nulog_model_latest.pt",
            "vocab_file": "vocab.txt",
        },
    }
    encoded_nulog_json = json.dumps(nulog_payload).encode()
    await nw.connect()
    await nw.publish(nats_subject="model_ready", payload_df=encoded_nulog_json)
    logging.info(
        "Published to model_ready Nats subject that new Nulog model is ready to be used for inferencing."
    )
    await nw.nc.close()


def main():
    loop = asyncio.get_event_loop()

    s3_client = boto3.resource(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    setup_task = loop.create_task(minio_setup_and_download_data(s3_client))
    if not loop.run_until_complete(setup_task):
        return

    train_task = loop.create_task(train_nulog_model(s3_client, "windows/"))
    if not loop.run_until_complete(train_task):
        return

    nats_signal_task = loop.create_task(send_signal_to_nats())
    loop.run_until_complete(nats_signal_task)

    loop.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.info(f"Nulog training failed. Exception {e}")
