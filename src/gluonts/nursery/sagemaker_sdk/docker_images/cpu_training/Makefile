ACCOUNT_ACCOUNT_ID=<yours, e.g. 123456654321>
PROFILE=<yours, e.g. default>
REPOSITORY=<yours, e.g. gluonts/latest_experiments>
REGION=<yours, e.g. us-west-2>
TAG=<yours, e.g. with-gluonts-cpu-latest>
ECR_IMAGE="$(ACCOUNT_ACCOUNT_ID).dkr.ecr.$(REGION).amazonaws.com/$(REPOSITORY):$(TAG)"

all:
	make dependencies
	make build
	make tag
	make login
	make push
	make clean

dependencies:
	@echo Copying dependencies.
	cp ../../../../../../requirements/requirements.txt .
	cp ../../../../../../requirements/requirements-extras-r.txt .
	cp ../../../../../../requirements/requirements-extras-prophet.txt .

#sagemaker_mxnet_container:
#	@echo Retrieving 'sagemaker-mxnet-container' dependency.
#	git clone https://github.com/aws/sagemaker-mxnet-container.git
#	cd sagemaker-mxnet-container
#	python setup.py sdist
#	cp dist/sagemaker-mxnet-container-*.tar.gz ../sagemaker_mxnet_container.tar.gz

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
	rm requirements.txt
	rm requirements-extras-r.txt
	rm requirements-extras-prophet.txt
#	rm -r sagemaker-mxnet-container