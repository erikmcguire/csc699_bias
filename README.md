# CSC 699: A Reproduction of [The Woman Worked as a Babysitter: On Biases in Language Generation](https://www.aclweb.org/anthology/D19-1339/)

An attempt for DePaul's CSC 699 (Spring 2020) at replicating a paper in a project inspired by [Noah Smith's NLP course](https://docs.google.com/document/d/1Dd9_VQHXseiroirUI-1rBDS6mJEUHiDQ7ND321O29W8/edit). Intended as a learning exercise, thus mistakes and misconceptions should be assumed to be our own.

* [Course presentation video](https://drive.google.com/file/d/16DQ3D1_TetnqQ_7a3_9KnbJNNvyGzTTV/view?usp=sharing)
* [Write-up](https://github.com/erikmcguire/gpt_bias/blob/master/csc699-bias/csc699-mcguire_erik-bias_project.pdf)

* [Main notebook on Colab](https://colab.research.google.com/drive/18KPRhuuUYMoLRZLRRgaI4XNJQT14bU-R?usp=sharing)
* [Data notebook on Colab](https://colab.research.google.com/drive/1qsgGGgQ0iSnumvQV0tp7XWu1CudNajEz?usp=sharing)

## Abstract

This is a reproduction of a previous work ([Sheng et al., 2019](https://arxiv.org/abs/1909.01326)) which studied the biases in text generated by language models. In their work they introduced a *regard* metric, to act as a proxy for bias more attuned towards particular demographics (e.g., *man* or *woman*) than conventional sentiment analysis. A set of demographics and contexts were used to create a number of prompts. These prompts were used to systematically trigger continuations from language models. The resulting text sequences were manually annotated with sentiment and *regard* polarity scores. Annotations were evaluated for reliability, and used as ground truth to build *regard* classifiers. Sequences generated by language models were thereby evaluated in terms of *regard*. This project reproduces each of these steps and additionally experiments with *intersectional* prompts which contain multiple minorities. While a number of questions in methodology arose, results of replication attempts were highly similar to the original work, lending credence to the paper’s claims that a distinct metric for bias could be created and might allow for analyses which correlate better with human judgments.
