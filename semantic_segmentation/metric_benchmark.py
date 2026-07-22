import os
import sys
import argparse

# Read --metric argument
parser = argparse.ArgumentParser(description='Run benchmark with different distance metrics', add_help=False)
parser.add_argument('--metric', type=str, default=None, choices=['cosine', 'l1', 'l2', 'jaccard', 'minkowski'])


# --dataset argument
parser.add_argument('--dataset', type=str, default=None, choices=['ade20k', 'voc12aug'])

# --epsilon argument
parser.add_argument('--epsilon', type=int, default=None, choices=[1, 2, 4, 6, 8, 12, 16])

# --iterations argument
parser.add_argument('--iterations', type=int, default=None, choices=[1, 3, 5, 10, 20, 40])

# --model argument
parser.add_argument('--model', type=str, default=None, choices=['deeplabv3', 'segformer', 'upernet', 'mask2former'])

# --backbone argument
parser.add_argument('--backbone', type=str, default=None, choices=['resnet50', 'mit-b0', 'mit-b1', 'mit-b2', 'mit-b3', 'mit-b4', 'mit-b5', 'internimage-t', 'internimage-s', 'internimage-b', 'internimage-h'])

# read all parse arguments
args, _ = parser.parse_known_args()




# Get metric value
metric = args.metric if args.metric is not None else os.environ.get('METRIC', 'cosine')
print(f'Running benchmark with metric: {metric}')


# Get dataset
dataset = args.dataset if args.dataset is not None else os.environ.get('DATASET', 'ade20k')


# Get epsilon value
epsilon = (args.epsilon / 255.0) if args.epsilon is not None else (float(os.environ.get('EPSILON', 8)) / 255.0)

# Get iterations amount
iterations = args.iterations if args.iterations is not None else int(os.environ.get('ITERATIONS', 20))

# Get modelencoder_decoder.py
model = args.model if args.model is not None else os.environ.get('MODEL', 'deeplabv3')

# Get backbone
backbone = args.backbone if args.backbone is not None else os.environ.get('BACKBONE', 'resnet50')


# Clear sys.argv so mmsegmentation does not see --metric and crash
sys.argv = [sys.argv[0]]

# Import and run evaluate safely
from semsegbench.evals import evaluate
model_used, results = evaluate(
    model_name=model,
    backbone=backbone,
    dataset=dataset,
    retrieve_existing=False,
    threat_config='configs/threat/adv_attacks.yml',
    metric=metric,
    epsilon=epsilon,
    iterations=iterations,
)

print(f'Model: {model}, Backbone: {backbone}, Dataset: {dataset}')
print(f'Metric used: {metric}, Epsilon: {epsilon}, Iterations:{iterations}')
print('Evaluation completed successfully.')
print('Results:', results)
