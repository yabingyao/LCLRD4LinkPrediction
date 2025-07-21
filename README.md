# LCLRD4LinkPrediction
## Abstract
Link prediction is widely used in various fields to
predict the missing or potential future links between node pairs
in the network. Knowledge distillation (KD) methods have been
introduced for the link prediction task, demonstrating exceptional
performance in inference acceleration. However, these methods
primarily focus on supervised learning settings that rely on highquality labels, and they fail to capture the rich structural features
inherent in graph data within the teacher model effectively,
resulting in suboptimal performance. To address the problem
of reducing label dependency, we propose a Link prediction
method via Contrastive Learning with Relational Distillation
(LCLRD) that aims to improve model performance under label scarcity. LCLRD incorporates contrastive learning into the
teacher model for pre-training, enabling the teacher model to
obtain high-quality node representations more effectively. Then,
graph structural information is extracted from the teacher model
via relational distillation and transferred to the student model.
Additionally, we leverage the Pearson correlation coefficient
(PCC) as a new matching strategy to replace the Kullback-Leibler
(KL) divergence, aiming to alleviate the prediction discrepancy
between the student and the stronger teacher model. The experimental results show that LCLRD significantly outperforms
other baseline methods on twelve datasets, demonstrating the
superiority of this approach.
##  Method Overview
<img width="1013" height="616" alt="image" src="https://github.com/user-attachments/assets/2f4a2f47-0077-4f8b-9176-e6d4822d3663" />
 Figure： Overall framework of LCLRD for link prediction
 
##  Experimental setup
igb==0.1.0

numpy==1.23.5

ogb==1.3.6

scikit_learn==1.2.1

scikit_learn==1.2.0

torch==1.13.1

torch_cluster==1.6.0

torch_geometric==2.2.0

torch_sparse==0.6.16+pt113cu117

## Run Setting
Teacher GNN training. You can change "sage" to "mlp" to obtain supervised training results with MLP.

```python teacher.py --datasets=cora --encoder=sage```

Student MLP training. L_D and L_R indicate the weights for the distribution-based and rank-based matching KD, respectively.

python main.py --datasets=cora --LLP_D=1 --LLP_R=1 --True_label=1
