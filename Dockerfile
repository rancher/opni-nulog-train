FROM rancher/opni-python-base:3.8-torch

COPY ./nulog-train/ /app/
COPY ./models/nulog/ /app/
RUN chmod a+rwx -R /app
WORKDIR /app

CMD ["python", "train.py"]