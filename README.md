# LearningMatch
A Siamese neural network that has learned the match manifold, see paper for more infomation. The datasets used to train LearningMatch can be accessed on Zenodo (10.5281/zenodo.14773846).

In order to use LearningMatch please make sure you install the following key packages NumPy, Joblib, Typing, Pandas, PyTorch, and PyCBC. See the `requirements.txt` for more information on versions. 

The repository conatins the following folders:

1) LearningMatchExample - This folder contains: a script containing the LearningMatch Model described in the paper (Model.py), a script that enables efficient training of the batched datasets (Dataset.py), and a script to train the LearningMatch model (Train.py).

2) Paper - This folder contatins the results showcased in the orginal paper:

    a) ComparisonLossCurve - This folder contatins the python scripts used to create the plots that compared training loss curves of different LearningMatch models and datasets. 

    b) DatasetGeneration - This folder contains the python scripts used to create both datasets (Normal and Diffused). These were then combined using a juypter notebook. 

    c) DeeperModelLayers2Layers5 - Trained LearningMatch model with a different architecture.

    d) DeeperModelLayers3Layers5 - Trained LearningMatch model with a different architecture.

    e) DeeperModelLayers4Layers2 - Trained LearningMatch model with a different architecture.

    f) DeeperModelLayers4Layers3 - Trained LearningMatch model with a different architecture.

    g) DeeperModelLayers4Layers4 - Trained LearningMatch model with a different architecture.

    h) DeeperModelLayers4Layers5 - Trained LearningMatch model described in paper.

    i) DeeperModelLayers4Layers6 - Trained LearningMatch model with a different architecture.

    j) DeeperModelLayers5Layers5 - Trained LearningMatch model with a different architecture.

    k) Timings - This folder contatins the python script used to determine how fast LearningMatch is at computing the match. 

    l) TrainingDataset25000 - Trained LearningMatch model with a smaller training dataset.

    m) TrainingDataset250000 - Trained LearningMatch model with a smaller training dataset.

    n) TrainingDataset2500000 - Trained LearningMatch model described in paper. A Juypter notebook called `Histograms' is used to generate the plot comparing dataset generation methods (Diffused and Normal). This trained LearningMatch model was also used to generate the various plots seen in the paper (Plot.py).
 


