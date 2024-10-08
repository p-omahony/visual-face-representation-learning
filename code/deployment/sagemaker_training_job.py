'''Launches a sagemaker training job.'''

import os

import boto3
import boto3.session
import sagemaker
from sagemaker.estimator import Estimator

# from dotenv import load_dotenv
# load_dotenv()

AWS_PROFILE_NAME = os.getenv('AWS_PROFILE_NAME')

IMAGE =  os.getenv('IMAGE_ECR_URI')
ROLE = os.getenv('SAGEMAKER_EXECUTION_ROLE')
DATA_LOCATION = os.getenv('SAGEMAKER_TJ_INPUT_DATA_PATH')
OUTPUT_PATH = os.getenv('SAGEMAKER_TJ_OUTPUT_PATH')

if __name__ == '__main__':
    boto_session = boto3.session.Session()
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    model = Estimator(
        IMAGE,
        ROLE,
        1,
        'ml.g4dn.xlarge',
        output_path=OUTPUT_PATH,
        sagemaker_session=sagemaker_session
    )

    model.fit(DATA_LOCATION, wait=False)