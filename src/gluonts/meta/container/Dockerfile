FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/mxnet-training:1.9.0-cpu-py38-ubuntu20.04-ec2

# we need pyarrow for parquet support, orjson for improved json performance
RUN pip3 install pyarrow==9.* orjson==3.8.*

RUN pip3 install gluonts[shell]==0.11.*

ENTRYPOINT ["python3", "-m", "gluonts.shell"]
