# GluonTS SageMaker SDK Tutorial

***This notebook is meant to be uploaded to a SageMaker notebook instance and executed there. As a kernel choose `conda_mxnet_p36`***

***In this how-to tutorial we will train a SimpleFeedForwardEstimator on the m4_hourly dataset on AWS SageMaker using the GluonTSFramework, and later review its performance. At the very end you will see how to launch your custom training script.*** <br/>
***In the end you should know how to train any GluonEstimator on any Dataset on SageMaker using the GluonTSFramework train(...) method, and how to run your own script using the run(...) method.***

## Notebook Setup

Currently, *GluonTSFramework* is only available through the master branch of *GluonTS*, so we install it with the required dependencies first:


```python
!pip install --upgrade mxnet==1.6  git+https://github.com/awslabs/gluon-ts.git#egg=gluonts[dev]
```


```python
# Third-party requirements
import boto3
import sagemaker
from pathlib import Path
import tempfile

# First-party requirements
from gluonts.nursery.sagemaker_sdk.estimator import GluonTSFramework
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.trainer import Trainer
```

## Credentials & Configuration

Since we are executing this tutorial on a SageMaker notebook instance, many parameters that we would usually need to predefine manually we can just retrieve from the environment. In order to highlight how you would have to set these parameters when you are executing a notebook like this on you local machine take a look at the cell output:


```python
temp_session = boto3.session.Session()
temp_sagemaker_session =  sagemaker.session.Session(boto_session=temp_session)
bucket_name = f"s3://{temp_sagemaker_session.default_bucket()}"
print(f"bucket_name = '{bucket_name}'")
region_name = temp_session.region_name
print(f"region_name = '{region_name}'")
profile_name = temp_session.profile_name
print(f"profile_name = '{profile_name}'")
iam_role = sagemaker.get_execution_role()
print(f"iam_role = '{iam_role}'")
```

Remember that in order to be able to use the profile 'defult' (or any other profile) on your local machine you must have correctly set up your [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html). Additionally, the specified bucket needs to be actually present in the specified region. With this out of the way, we can continue as if we had set the above variables manually.

## Experimental Setup

### Experiment directory

First, we should define the *S3 parent folder location* which will later contain the folder with all the data generated during the experiment (model artifacts, custom scripts, dependencies etc.). I you choose to use a subfolder for your experiments (like we do here) the folder does not have to exist yet, but it's name must satisfy the regular expression pattern: \^\[a-zA-Z0-9\](-\*\[a-zA-Z0-9\])\*. If not specified, the default bucket of the specified region itself will be used.


```python
experiment_parent_dir = bucket_name + "/my-sagemaker-experiments"
print(f"experiment_parent_dir = '{experiment_parent_dir}'")
```

### SageMaker session

Next, we need to create a sagemaker session in our region using a [*boto3*](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#using-boto-3) session with our credentials (profile).


```python
boto_session = boto3.session.Session(profile_name=profile_name, region_name=region_name)
sagemaker_session =  sagemaker.session.Session(boto_session=boto_session)
```

### AWS IAM role

We also need to provide an AWS [IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html) role, with which to access the resources on our account.


```python
role = iam_role
```

### Training image & instance type

We can just use one of the prebuilt SageMaker [ECR](https://docs.aws.amazon.com/AmazonECR/latest/userguide/docker-basics.html) images and install the gluonts version we prefer dynamically though the 'requirements.txt'.


```python
general_instance_type = "cpu" 
# instance_type = "gpu" # alternative
```

Depending our *general_instance_type* choice we will have to select an appropriate concrete 'instance type':


```python
instance_type = "ml.c5.xlarge" if general_instance_type == "cpu" else "ml.p2.xlarge" 
```

and an appropriate prebuilt mxnet image (we will take the training images here):


```python
if general_instance_type == "cpu":
    docker_image = f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-cpu-py36-ubuntu16.04"
else:
    docker_image = f"763104351884.dkr.ecr.{region_name}.amazonaws.com/mxnet-training:1.6.0-gpu-py36-cu101-ubuntu16.04"
print(f"docker_image = '{docker_image}'")
```

### Base job description

We can give our training job a base name that lets us easily identify experiments of the same type. <br/>
It has to satisfy the regular expression pattern: \^\[a-zA-Z0-9\](-\*\[a-zA-Z0-9\])\*


```python
base_job_description = "my-sagemaker-experiment-intro"
```

### Dataset

Here we have two choices; we can either pick a built in dataset provided by GluonTS or any dataset in the GluonTS dataset format located on S3, which would look like this:

>dataset_name<br/>
>&nbsp;&nbsp;&nbsp;&nbsp;|---> train<br/>
>&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|--> data.json<br/>
>&nbsp;&nbsp;&nbsp;&nbsp;|---> test<br/>
>&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|--> data.json<br/>
>&nbsp;&nbsp;&nbsp;&nbsp;|---> metadata.json<br/>

Since we haven't uploaded any, lets pick a provided one for now. <br/>
The following datasets are available:


```python
print(dataset_recipes.keys())
```

How about "m4_hourly"?:


```python
dataset_name = "m4_hourly"
# dataset_name = "s3://<your-custom-dataset-location>" # if using a custom dataset
```

We will need to know the *prediction_length* and *freq* of the dataset to define our SimpleFeedForwardEstimator, so lets keep track of them:


```python
freq = dataset_recipes[dataset_name].keywords["pandas_freq"]
prediction_length = dataset_recipes[dataset_name].keywords["prediction_length"]
```

### Requirements and Dependencies

We will additionally have to specify a 'requirements.txt' file where we specify which GluonTS version we want to use. <br/>
Here we will create a temporary requirements file, but you can just have a 'requirements.txt' file in the folder where you launch your experiments.


```python
requirements_dot_txt_file_name = "requirements.txt"
requirements_dot_txt_file_content = "git+https://github.com/awslabs/gluon-ts.git"
```


```python
# only using temporary directory for demonstration
temp_dir = tempfile.TemporaryDirectory()
temp_dir_path = Path(temp_dir.name)

# create the requirements.txt file
with open(temp_dir_path / requirements_dot_txt_file_name, "w") as req_file: # has to be called requirements.txt
    req_file.write(requirements_dot_txt_file_content)
my_requirements_txt_file_path = str(temp_dir_path / requirements_dot_txt_file_name)
print(f"my_requirements_txt_file_path = '{my_requirements_txt_file_path}'")
```

### Define the Estimator

Now we define the Estimator we want to train, which can be any GluonEstimator (except ) with any hyperparameter.


```python
my_estimator = SimpleFeedForwardEstimator(
                    prediction_length=prediction_length,
                    freq=freq,
                    trainer=Trainer(ctx=general_instance_type, epochs=5) # optional
                )
```

## Launch the Experiment


```python
my_experiment = GluonTSFramework(
                    sagemaker_session=sagemaker_session,
                    role=role,
                    image_name=docker_image,  
                    base_job_name=base_job_description,
                    train_instance_type=instance_type,
                    dependencies=[my_requirements_txt_file_path],
                    output_path=experiment_parent_dir, # optional, but recommended
                    code_location=experiment_parent_dir, # optional, but recommended
                )
```

And finally we call the *train* method to train our estimator, where we just specify our dataset and estimator:



```python
results = my_experiment.train(dataset=dataset_name, estimator=my_estimator) 
```

## Review the Results

The 'train(...)' function returnes a TrainResult which consists of the following fields:


```python
print(results._fields)
```

So we could use the predictor straight away to predict on some additional data if we would like. <br/>
We can also inspect our training history and monitored metrics (like resource consumption or epoch loss) on SageMaker under "Training/Training jobs" here:


```python
print(f"https://{region_name}.console.aws.amazon.com/sagemaker/home?region={region_name}#/jobs/{results.job_name}")
```

Or take a look at the metrics right here:


```python
results.metrics[0]
```

Or head to our bucket to download the model artifacts:


```python
print(f"https://s3.console.aws.amazon.com/s3/buckets/{experiment_parent_dir[5:]}/{results.job_name}/?region={region_name}&tab=overview")
```

## Run a custom python script

There process to run a custom python script is not much different, however, you will have to adapt your usual python script to particularities of the SageMaker.


```python
import os
import gluonts
import s3fs
```

### Writing a custom script

Your custom script has to adhere to a rough format, for this reason we provide the "run_entry_point.py" script with GluonTS under:


```python
run_entry_point_path = (
    Path(os.path.dirname(gluonts.__file__))
    / "nursery"
    / "sagemaker_sdk"
    / "entry_point_scripts"
    / "run_entry_point.py"
)
```

Lets take a look:


```python
with open(run_entry_point_path, 'r') as script:
    print(script.read())
```

As we can see, there is a *run* method, whithin which we are supposed to write our custom code.

Additionally, at the bottom we might need to parse additional arguments that we provide for example through the "inputs" parameter of the GluonTSFramework.run(...) method. The "inputs" parameter cannot be empty, due to the restrictions of the Framework baseclass of the GluonTSFramework, however, you can pass an empty file located on S3 as dummy input.

Lets define a path for the dummy file:


```python
dummy_s3_file_path = bucket_name + "/dummy_1234"
print(f"dummy_s3_file_path = '{dummy_s3_file_path}'")
```

Lets create the S3 file (if the file already exists you will have to set overwrite to 'True', or choose a different path for the dummy file):


```python
overwrite = False
s3 = s3fs.S3FileSystem(anon=False)  # uses default credentials
if not(s3.exists(dummy_s3_file_path)) or overwrite:
    with s3.open(dummy_s3_file_path, 'w') as f:
        f.write("This is a dummy file.")  
    print("Dummy file created!")
else:
    print("No dummy file created!")
```


```python
my_inputs = {'my_dataset_name': sagemaker.s3_input(dummy_s3_file_path, content_type='application/json')} 
```

If we were to pass a dataset location as input as defined above, we would have to parse the location of that dataset (which will be uploaded into the container environment) for example like this:

> parser.add_argument('--my_fancy_dataset', type=str, default=os.environ['SM_CHANNEL_MY_DATASET_NAME'])

Prepending "SM_CHANNEL_" and converting the name to all caps is necessary. <br/>
Within the *run(...)* method the location will be accessible by:

> arguments.my_fancy_dataset

Any additional "hyperparameter" you provide to *GluonTSFramework.run(...)* are already parsed by:

>parser.add_argument("--sm-hps", type=json.loads, default=os.environ["SM_HPS"])

#### Get familiar tasks:

For now, we will only use the unmodified run script, however, a good exercise to get familiar with the framework would be to modify the script so:
* You parse the location of the input we provide thourgh "my_inputs" 
* You read the dummy file inside the run(...) method
* You write the content of the file to a new file called "parsed.txt" and save it to the output location 
* You check in S3 that "parsed.txt" was saved to S3 in your experiment folder under /output/output.tar.gz

HINT: you don't need to write or read form S3 explicitly, but rather access the appropriate local location through "arguments" of the run(...) method within your scripts; let SageMaker containers handle the interaction with S3. <br/>
HINT: you can take a look at the "train_entry_point.py" to see an actual example for a training script.

### Run the Experiment

As we will see, the arguments to the GluonTSFramework run(...) method are almost identical to the train(...) one, however, we additionally specify the required "entry_point" and "inputs", and optionally "wait=False" because we might want to launch multiple jobs async.


```python
my_experiment, my_job_name = GluonTSFramework.run(
                    entry_point=str(run_entry_point_path), # additionally required
                    inputs = my_inputs, # additionally required
                    sagemaker_session=sagemaker_session,
                    role=role,
                    image_name=docker_image,  
                    base_job_name=base_job_description,
                    train_instance_type=instance_type,
                    dependencies=[my_requirements_txt_file_path],
                    output_path=experiment_parent_dir, # optional, but recommended
                    code_location=experiment_parent_dir, # optional, but recommended
                    wait=False # optional
                )
```

We can take a look at the training job right away:


```python
print(f"https://{region_name}.console.aws.amazon.com/sagemaker/home?region={region_name}#/jobs/{my_job_name}")
```

And again, check out the corresponding S3 location:


```python
print(f"https://s3.console.aws.amazon.com/s3/buckets/{experiment_parent_dir[5:]}/{my_job_name}/?region={region_name}&tab=overview")
```

### Custom GluonTS version:


In case you are modifying GluonTS on your local machine and want to run experiments on your custom version, just import GluonTS and define:

>gluont_ts_path = Path(gluonts.__path__[0]) <br/>
>gluont_ts_requirements_path = gluont_ts_path.parent.parent / "requirements" / "requirements.txt"

and change the dependencies argument of run(...) or train(...) the following way:

> dependencies=[gluont_ts_requirements_path, gluont_ts_path]

## Cleanup

Lets just clean up the temporary directory:


```python
temp_dir.cleanup()
```


```python

```
