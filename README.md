## Nulog Training Job

* This job will train a Nulog model when the training controller service receives a signal from NATS.
  * Pre-requisite: An NVIDIA GPU driver must be installed as it is required for training.
    ```
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml
    The training controller service should also be running beforehand as well as should NATS, Traefik and Elasticsearch.
    ```
* The job will train a Nulog model based on logs retrieved from Elasticsearch.
*To test this service, you can send the payload below to the NATS subject "train". The training controller must be running as it will pick up and start the training job.
  ```
  payload = {"model_to_train": "nulog","time_intervals": [{"start_ts": TIMESTAMP_NANO, "end_ts": TIMESTAMP_NANO}, {"start_ts": TIMESTAMP_NANO, "end_ts": TIMESTAMP_NANO}, ...]}
  ```

## Contributing
We use `pre-commit` for formatting auto-linting and checking import. Please refer to [installation](https://pre-commit.com/#installation) to install the pre-commit or run `pip install pre-commit`. Then you can activate it for this repo. Once it's activated, it will lint and format the code when you make a git commit. It makes changes in place. If the code is modified during the reformatting, it needs to be staged manually.

```
# Install
pip install pre-commit

# Install the git commit hook to invoke automatically every time you do "git commit"
pre-commit install

# (Optional)Manually run against all files
pre-commit run --all-files
```
