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

MINIO_SERVER_URL = os.environ["MINIO_SERVER_URL"]
MINIO_ACCESS_KEY = os.environ["MINIO_ACCESS_KEY"]
MINIO_SECRET_KEY = os.environ["MINIO_SECRET_KEY"]

nw = NatsWrapper()


def train_nulog_model(minio_client, windows_folder_path):
    nr_epochs = 10
    num_samples = 0
    parser = LogParser()
    train_texts, validation_texts = parser.load_data(windows_folder_path)
    train_tokenized = parser.tokenize_data(train_texts, isTrain=True)
    validation_tokenized = parser.tokenize_data(validation_texts, isTrain=False)
    parser.tokenizer.save_vocab()
    parser.train(
        train_tokenized,
        validation_tokenized,
        nr_epochs=nr_epochs,
        num_samples=num_samples,
    )
    all_files = os.listdir("output/")
    if "nulog_model_latest.pt" in all_files and "vocab.txt" in all_files:
        logging.info("Completed training model")
        minio_client.meta.client.upload_file(
            "output/nulog_model_latest.pt", "nulog-models", "nulog_model_latest.pt"
        )
        minio_client.meta.client.upload_file(
            "output/vocab.txt", "nulog-models", "vocab.txt"
        )
        logging.info("Nulog model and vocab have been uploaded to Minio.")
        os.remove("output/nulog_model_latest.pt")
        os.remove("output/vocab.txt")

    else:
        logging.info("Nulog model was not able to be trained and saved successfully.")
        return False
    return True


def minio_setup_and_download_data(minio_client):
    try:
        minio_client.meta.client.download_file(
            "training-logs", "windows.tar.gz", "windows.tar.gz"
        )
        logging.info("Downloaded logs from minio successfully")

        shutil.unpack_archive("windows.tar.gz", format="gztar")
    except EndpointConnectionError:
        logging.error(
            f"Could not connect to minio with endpoint_url={MINIO_SERVER_URL} aws_access_key_id={MINIO_ACCESS_KEY} aws_secret_access_key={MINIO_SECRET_KEY} "
        )
        logging.info("Job failed")
        return False

    try:
        minio_client.meta.client.head_bucket(Bucket="nulog-models")
        logging.info("nulog-models bucket exists")
    except botocore.exceptions.ClientError as e:
        # If a client error is thrown, then check that it was a 404 error.
        # If it was a 404 error, then the bucket does not exist.
        error_code = e.response["Error"]["Code"]
        if error_code == "404":
            logging.info("nulog-models bucket does not exist so creating it now")
            minio_client.create_bucket(Bucket="nulog-models")
    return True


async def send_signal_to_nats():

    nulog_payload = {
        "bucket": "nulog-models",
        "bucket_files": {
            "model_file": "nulog_model_latest.pt",
            "vocab_file": "vocab.txt",
        },
    }
    encoded_nulog_json = json.dumps(nulog_payload).encode()
    await nw.publish(nats_subject="model_ready", payload_df=encoded_nulog_json)
    logging.info(
        "Published to model_ready Nats subject that new Nulog model is ready to be used for inferencing."
    )


async def consume_signal(job_queue):
    await nw.subscribe(
        nats_subject="gpu_service_training_internal", payload_queue=job_queue
    )


async def train_model(job_queue):
    minio_client = boto3.resource(
        "s3",
        endpoint_url=MINIO_SERVER_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )

    windows_folder_path = "windows/"

    while True:
        new_job = await job_queue.get()

        res_download_data = minio_setup_and_download_data(minio_client)
        res_train_model = train_nulog_model(minio_client, windows_folder_path)
        await send_signal_to_nats()


async def init_nats():
    logging.info("Attempting to connect to NATS")
    await nw.connect()


def main():
    loop = asyncio.get_event_loop()
    job_queue = asyncio.Queue(loop=loop)

    consumer_coroutine = consume_signal(job_queue)
    training_coroutine = train_model(job_queue)

    task = loop.create_task(init_nats())
    loop.run_until_complete(task)

    loop.run_until_complete(asyncio.gather(training_coroutine, consumer_coroutine))
    try:
        loop.run_forever()
    finally:
        loop.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.info(f"Nulog training failed. Exception {e}")
