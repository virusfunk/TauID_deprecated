# Tau IDentification(TauID) using Neural Network

## Introduction
* This project aims to implement identification for tau decay.
* 5 haronic decay modes(Z->tautau->hadronic) and a main background mode(Z->qqbaar) are considered, which have high Branching Ratio.
    * Detail 1 : [PDG/Z Boson](https://pdg.lbl.gov/2018/listings/rpp2018-list-z-boson.pdf)
    * Detail 2 : [PDG/tau Branching Fractions](https://pdg.lbl.gov/2011/reviews/rpp2011-rev-tau-branching-fractions.pdf)

## Environment
* CUDA 11.1 & cuDNN 8.
* Tensorflow nightly version(>=2.5) is used because Nvidia RTX 3090 needs cuDNN 8.
* ROOT 6.24.00 is used.
    * ROOT 6.20.04 seems to have a bug in loading data to `numpy.ndarray`.


## Dataset

* Image shape = (256x256x4)
* train : validation : test = 105480 : 19800 : 18000
* Datasets were generated via HTcondor in KCMS cluster. 
    * About 1500 * 50 events are generated per decay channels.
    * Generated datasets are located in UOS server ; `/store/ml/dual-readout/TauID/hadronic/`.
* One hot encoding is applied for labelling.


* Detail : [purol/dual-readout repo.](https://github.com/purol/dual-readout) 

## Model available

* Vanila Convolutional Neural Network (CNN)
* Vision Transformer (ViT)


