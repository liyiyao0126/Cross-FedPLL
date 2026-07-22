# Dual History-Based Mitigation of Label Noise for Federated Partial-Label Learning

## Abstract
Federated Partial-Label Learning (FedPLL) trains models on decentralized data where instances are annotated with multiple ambiguous labels. However, accurately identifying true labels from such candidate label sets is critical yet challenging. Compounding this challenge, complex label noise patterns are often intertwined in practical scenarios. Local model, trained with limited instances, is vulnerable to erroneous labels. Global model struggles to distinguish between personalized shift labels and erroneous ones. In this paper, we propose a novel framework Cross-FedPLL, which fully leverages the historical predictions from the global and local models for accurate label identification. Firstly, we design a cross selection strategy enabling mutual high-confidence label selection between the global and local models. Besides, to tackle data sparsity caused by cross selection, we propose a confidence-aware data augmentation method. Furthermore, to avoid misfiltering personalized shift true labels, we employ expertise-weighted Kullback-Leibler divergence for personalization mining. Results demonstrate that Cross-FedPLL outperforms the baselines by an average of 19.36\% in the complex label noise scenario.

## Train Command

python main.py --dataset cifar10 --batch-size 64 --lr 0.0001 --seed 5 --out cifar10@q03@c10@resnet@a002@p02 --partial_rate 0.3 --epochs 30 --sharpen_T 0.5 --use_mix --lambda_cr 4 --num_rounds 30 --num_clients 10 --select_threshold 0.8 --warm_up 10 --annotation_ratio 0.02 --drift_rate 0.2 --arch resnet18



