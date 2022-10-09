FROM nvcr.io/nvidia/pytorch:21.06-py3

RUN apt update && DEBIAN_FRONTEND=noninteractive \
    apt install libsndfile1-dev ffmpeg -y && \
    apt-get clean && \
	rm -rf /var/lib/apt/lists/*

RUN pip install llvmlite --ignore-installed

ADD requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ADD . /app
WORKDIR /app

ENTRYPOINT ["python", "transcribe.py"]
CMD ["/input", "--output-folder", "/output"]
