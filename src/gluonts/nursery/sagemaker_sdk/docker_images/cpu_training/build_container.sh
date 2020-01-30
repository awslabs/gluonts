#!/bin/bash
set -e
set -x

ACCOUNT_ID=670864377759
IMAGE_NAME="gluonts/sagemaker_api"
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.us-west-2.amazonaws.com/$IMAGE_NAME:1.4.1-0.4.1-gluonts-cpu-latest"

docker build -t "$IMAGE_NAME:latest" .
$(aws ecr get-login --region us-west-2 --no-include-email --registry-ids "$ACCOUNT_ID" --profile default)
docker tag "$IMAGE_NAME:latest" "$ECR_IMAGE"
docker push "$ECR_IMAGE"
