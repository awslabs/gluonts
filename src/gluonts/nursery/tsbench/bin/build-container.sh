#!/bin/bash
set -e

echo "Building image..."
docker build \
    -t $REGISTRY/ts-bench:latest \
    -f Dockerfile .

echo "Fetching credentials from AWS..."
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region)
ECR_PASSWORD=$(aws ecr get-login-password)
REGISTRY=$AWS_ACCOUNT.dkr.ecr.$AWS_REGION.amazonaws.com

echo "Logging in to ECR..."
echo $ECR_PASSWORD | \
    docker login --username AWS --password-stdin $REGISTRY

echo "Pushing image..."
docker push $REGISTRY/tsbench:latest
