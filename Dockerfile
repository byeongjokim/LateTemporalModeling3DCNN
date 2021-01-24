FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
COPY . /app
WORKDIR /app
RUN pip install -r r.txt
ENTRYPOINT ["python"]
CMD ["two_stream_bert2.py"]