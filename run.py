import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::899212678931:role/service-role/AmazonSageMaker-ExecutionRole-20260405T024066"
checkpoint_s3_uri = 's3://agression-model/vivit/checkpoints/'
output_s3_uri = 's3://agression-model/vivit/'

estimator = PyTorch(
    source_dir="./src",
    entry_point='train.py',
    role=role,
    #use_spot_instances=True,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version ='2.5.1',
    py_version='py311',
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },
    hyperparameters={
        'epochs': 25,
        'batch-size': 64,
        'checkpoint-dir': '/opt/ml/checkpoints/',
        'model-dir': '/opt/ml/model',
        'data-dir': '/opt/ml/input',
        'threshold': 0.3,
    },
    output_path=output_s3_uri,
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path='/opt/ml/checkpoints/',
)

estimator.fit()