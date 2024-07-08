In this repository you will find two folders:
- LinearInterpolation
- DeepEnsemble

These include the scripts from our paper "How and why does deep ensemble coupled with transfer learning increase performance in bipolar disorder and schizophrenia classification?" that can be found here : https://hal.science/hal-04631924/document 

The "LinearInterpolation" repository contains the scripts used to study the loss landscape of models trained to classify bipolar disorder (BD) or schizophrenia (SCZ) from healthy controls (HC) when they were trained from pre-trained weights VS randomly-initialized weights (for more information on the classifiers used : https://github.com/Duplums/SMLvsDL).

We used the method proposed in "What is being transferred in transfer learning?" (https://arxiv.org/abs/2008.11687), and applied it to densenet121 models trained to classify anatomical MRIs of HC from MRIs of BD or SCZ subjects.

The "DeepEnsemble" repository contains the scripts used to study the ideal number of models to use to get the best trade-off in terms of gains of performance and computational requirements.
