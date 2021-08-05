FROM python:3.8-slim as base

FROM base as builder

RUN apt-get update \
    && apt-get install gcc -y \
    && apt-get clean

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./nulog-train/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

FROM base

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./nulog-train/ /app/
COPY ./models/nulog/ /app/
RUN chmod a+rwx -R /app
WORKDIR /app

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

CMD ["python", "train.py"]
