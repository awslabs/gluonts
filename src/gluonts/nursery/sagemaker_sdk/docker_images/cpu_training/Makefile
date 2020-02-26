ACCOUNT_ACCOUNT_ID=<yours, e.g. 123456654321>
PROFILE=<yours, e.g. default>
REPOSITORY=<yours, e.g. gluonts/latest_experiments>
REGION=<yours, e.g. us-west-2>
TAG=<yours, e.g. with-gluonts-cpu-latest>
ECR_IMAGE="$(ACCOUNT_ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(REPOSITORY):$(TAG)"

all:
	make sagemaker-mxnet-container_dependency
	make build
	make tag
	make login
	make push
	make clean

sagemaker-mxnet-container_dependency:
	@echo Retrieving 'sagemaker-mxnet-container' dependency.
	git clone https://github.com/aws/sagemaker-mxnet-container.git
	cd sagemaker-mxnet-container; python setup.py sdist
	cp sagemaker-mxnet-container/dist/sagemaker_mxnet_*.tar.gz sagemaker_mxnet_container.tar.gz

build:
	@echo Making $(REPOSITORY) image.
	docker build -t "$(REPOSITORY):latest" .

tag:
	@echo Tagging base image for repository $(REPOSITORY) with $(TAG).
	docker tag "$(REPOSITORY):latest" $(ECR_IMAGE)

login:
	@echo Logging into AWS ECR.
	$(shell aws ecr get-login --region $(REGION) --no-include-email --registry-ids $(ACCOUNT_ACCOUNT_ID) --profile $(PROFILE))

push:
	@echo Pushing image $(ECR_IMAGE) to AWS ECR.
	docker push $(ECR_IMAGE)

clean:
	rm -rf sagemaker-mxnet-container
	rm sagemaker_mxnet_container.tar.gz
