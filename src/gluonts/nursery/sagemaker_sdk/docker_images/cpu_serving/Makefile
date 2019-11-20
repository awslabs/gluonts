ACCOUNT_ACCOUNT_ID=<yours, e.g. 123456654321>
PROFILE=<yours, e.g. default>
REPOSITORY=<yours, e.g. gluonts/latest_experiments>
REGION=<yours, e.g. us-west-2>
TAG=<yours, e.g. with-gluonts-cpu-latest>
ECR_IMAGE="$(ACCOUNT_ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(REPOSITORY):$(TAG)"

all:
	make gluonts_dependencies
	make sagemaker-mxnet-serving-container_dependency
	make build
	make tag
	make login
	make push
	make clean

gluonts_dependencies:
	@echo Copying dependencies.
	cp ../../../../../../requirements/requirements.txt .
	cp ../../../../../../requirements/requirements-extras-r.txt .
	cp ../../../../../../requirements/requirements-extras-prophet.txt .

sagemaker-mxnet-serving-container_dependency:
	@echo Retrieving 'sagemaker-mxnet-serving-container' dependency.
	git clone https://github.com/aws/sagemaker-mxnet-serving-container.git
	cd sagemaker-mxnet-serving-container; python setup.py sdist
	cp sagemaker-mxnet-serving-container/dist/sagemaker_mxnet_serving_container-*.tar.gz sagemaker_mxnet_serving_container.tar.gz
	cp sagemaker-mxnet-serving-container/docker/1.4.1/py3/config.properties .
	cp sagemaker-mxnet-serving-container/docker/1.4.1/py3/mms-entrypoint.py .

build:
	@echo Making $(REPOSITORY) image.
	docker build -t "$(REPOSITORY):latest" .

tag:
	@echo Tagging base image for repository $(REPOSITORY) with $(TAG).
	docker tag "$(REPOSITORY):latest" $(ECR_IMAGE)

login:
	@echo Logging into AWS ECR.
	aws ecr get-login --region $(REGION) --no-include-email --registry-ids $(ACCOUNT_ACCOUNT_ID) --profile $(PROFILE)

push:
	@echo Pushing image $(ECR_IMAGE) to AWS ECR.
	docker push $(ECR_IMAGE)

clean:
	rm requirements.txt
	rm requirements-extras-r.txt
	rm requirements-extras-prophet.txt
	rm -rf sagemaker-mxnet-serving-container
	rm sagemaker_mxnet_serving_container.tar.gz
	rm config.properties
	rm mms-entrypoint.py#	rm -r sagemaker-mxnet-container