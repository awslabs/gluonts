# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import logging
import time

# import tarfile
from pathlib import Path

# from keymaker import KeyMaker
import boto3
import botocore
import pandas as pd
import sagemaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
)
logger = logging.getLogger(__name__)
boto_session = boto3.Session(region_name="us-east-1")
# km = KeyMaker()
# region_name = "us-east-1"
sagemaker_session = sagemaker.Session(boto_session=boto_session)


def download(job_name, destination_dir, sagemaker_session, csv_file):
    df = pd.read_csv(csv_file)
    idx = df[df["TrainingJobName"] == jobname].index[0]
    DownloadStatus = bool(df["DownloadStatus"][idx])
    TrainingJobStatus = df["TrainingJobStatus"][idx]
    if DownloadStatus:
        logger.info(f"{job_name} has already been downloaded")
    elif TrainingJobStatus in ["Failed", "Stopped"]:
        logger.info(
            f"{jobname} has already been registered as {TrainingJobStatus} so we won't query it anymore."
        )
    else:
        attempt_description = 0
        NB_ATTEMPTS_DESCRIPTION = 5
        for _ in range(NB_ATTEMPTS_DESCRIPTION):
            try:
                json_file = sagemaker_session.describe_training_job(
                    job_name=job_name
                )
            except botocore.exceptions.ClientError as e:
                attempt_description += 1
                logger.warning(f"Exception {e} has been raised.")
                logger.warning(
                    f"Retrying to describe {job_name}, {NB_ATTEMPTS_DESCRIPTION - attempt_description} attemps left for this resource"
                )
                logger.warning("Sleeping two seconds first")
                time.sleep(2)
            else:
                break
        else:
            logger.warning(
                f"All the attempts have failed to describe {jobname}."
            )
        TrainingJobStatus = json_file["TrainingJobStatus"]
        TrainingJobName = json_file["TrainingJobName"]
        df.loc[idx, "TrainingJobStatus"] = TrainingJobStatus
        logger.info(
            f"Job: {TrainingJobName}, Status: {TrainingJobStatus}, Downloaded : {DownloadStatus}"
        )
        if TrainingJobStatus == "Completed":
            if DownloadStatus is False:
                s3 = boto_session.resource("s3")
                destination_dir = Path(destination_dir)
                destination_dir.mkdir(parents=True, exist_ok=True)
                output_path = destination_dir / f"{TrainingJobName}.tar.gz"
                logger.info(
                    f"Downloading {TrainingJobName} in directory {destination_dir}"
                )
                attempts = 0
                NB_ATTEMPTS = 5
                for attempt in range(NB_ATTEMPTS):
                    try:
                        s3.Bucket("s3-violation-test-bucket").download_file(
                            str(
                                Path(
                                    TrainingJobName, "output", "output.tar.gz"
                                )
                            ),
                            str(output_path),
                        )
                    except botocore.exceptions.ClientError as e:
                        attempts += 1
                        logger.warning(f"Exception {e} has been raised.")
                        logger.warning(
                            f"Retrying to download {TrainingJobName}, {NB_ATTEMPTS - attempts} attemps left for this resource"
                        )
                        logger.warning("Sleeping two seconds first")
                        time.sleep(2)
                    else:
                        logger.info(
                            f"{TrainingJobName} has been succesfully downloaded"
                        )
                        break
                else:
                    logger.warning(
                        f"All the attempts have failed to download {TrainingJobName}."
                    )
                df.loc[idx, "DownloadStatus"] = True
                df.loc[idx, "SaveLocation"] = str(destination_dir)
        df.to_csv(csv_file, index=False)


csv_file_path = "/home/ec2-user/SageMaker/gluon-ts-gan/src/gluonts/model/syntheticTransformer/TrainingFollowUp.csv"
df = pd.read_csv(csv_file_path)

list_jobs = list(df["TrainingJobName"])

destination = Path("/home/ec2-user/SageMaker/TrainedModels")
destination.mkdir(parents=True, exist_ok=True)
# csv_file = csv_file_path
# download(
#     "PSA-GAN-electricity-len64-27082021-004852-508",
#     destination,
#     sagemaker_session,
#     csv_file,
# )

for k in range(len(list_jobs)):
    jobname = list_jobs[k]
    #     print(jobname)
    download(jobname, destination, sagemaker_session, csv_file_path)
