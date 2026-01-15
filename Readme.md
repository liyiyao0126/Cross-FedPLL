# Dual Histories Can Assist: Cross-Selecting Global Confident Labels and Mining Personalized Shifts in Federated Partial-Label Learning

## Abstract
Federated Partial-Label Learning (FedPLL) integrates partial-label learning (PLL) with federated learning (FL) to handle decentralized data where training instances are annotated with multiple candidate labels while preserving data privacy. Identifying the true labels accurately and completely is critical yet challenging. Local PLL model, trained with limited instances, is vulnerable to interference from incorrect labels, making accurate discrimination difficult. Global model, integrating multi-party knowledge, excels at identifying global confident labels, yet struggles to distinguish between personalized shift labels and erroneous ones. In this paper, we propose a novel framework Cross-FedPLL, which leverages the historical predictions from the global and local models for accurate and holistic label identification. Firstly, we design a cross selection strategy enabling mutual high-confidence label selection between the global and local models to combat label noise. Besides, to tackle data sparsity from cross selection, we propose a confidence-aware data augmentation method. Furthermore, to avoid misfiltering personalized shift true labels, we employ expertise-weighted Kullback-Leibler divergence for personalization mining. Experimental results demonstrate that Cross-FedPLL outperforms the state-of-the-art methods by 19.53% in the complex label scenario.


## Train Command

python main.py --dataset cifar10 --batch-size 64 --lr 0.0001 --seed 5 --out cifar10@q03@c10@resnet@a002@p02 --partial_rate 0.3 --epochs 30 --sharpen_T 0.5 --use_mix --lambda_cr 4 --num_rounds 30 --num_clients 10 --select_threshold 0.8 --warm_up 10 --annotation_ratio 0.02 --drift_rate 0.2 --arch resnet18



