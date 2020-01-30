#!/bin/bash
set -e
set -x

ACCOUNT_ID="<your-account-id>"
IMAGE_NAME="<some folder, for example: gluonts/own>"
PROFILE="<default or your-profile>"
REGION="<your-region>"
ECR_IMAGE="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$IMAGE_NAME:with-gluonts-cpu-latest"

docker build -t "$IMAGE_NAME:latest" .
$(aws ecr get-login --region "$REGION" --no-include-email --registry-ids "$ACCOUNT_ID" --profile "$PROFILE")
docker tag "$IMAGE_NAME:latest" "$ECR_IMAGE"
docker push "$ECR_IMAGE"
