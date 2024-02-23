In this repository you will find two folders:
- LinearInterpolation
- DeepEnsemble
These include the scripts that were used in our paper "How and why does deep ensemble coupled with transfer learning increase performance in bipolar disorder and schizophrenia classification?".

The "LinearInterpolation" repository contains the scripts used to study the loss landscape of our models when they were trained from pre-trained weights VS from randomly-initialized weights.
We used the method proposed in "What is being transferred in transfer learning?" (https://arxiv.org/abs/2008.11687), and applied it to densenet121 models trained to classify anatomical MRIs of healthy controls from MRIs of bipolar disorder (BD) or schizophrenia (SCZ) subjects.

The "DeepEnsemble" repository contains the scripts used to study the ideal number of models to use to get the best trade-off in terms of gains of performance and computational requirements.
