import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.inputs import TrainingInput

role = "arn:aws:iam::899212678931:role/service-role/AmazonSageMaker-ExecutionRole-20260405T024066"
checkpoint_s3_uri = 's3://agression-model/vivit/checkpoints/'
local_checkpoint_dir = '/opt/ml/checkpoints'

estimator = PyTorch(
    source_dir="./src",
    entry_point='train.py',
    role=role,
    #use_spot_instances=True,
    instance_type='ml.g4dn.xlarge',
    instance_count=1,
    framework_version='2.5.1',
    py_version='py311',
    distribution={
        'torch_distributed': {
            'enabled': True
        }
    },
    hyperparameters={
        'epochs': 10,
        'batch-size': 2,
        'checkpoint-dir': local_checkpoint_dir,
        'model-dir': '/opt/ml/model',
        'data-dir': '/opt/ml/input'
    },
    output_path='s3://agression-model/',
    checkpoint_s3_uri=checkpoint_s3_uri,
    checkpoint_local_path=local_checkpoint_dir,
)

estimator.fit()