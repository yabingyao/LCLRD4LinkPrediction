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
