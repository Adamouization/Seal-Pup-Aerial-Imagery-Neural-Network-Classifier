# Seal Pup Aerial Imagery Neural Network Classifier 

[![HitCount](http://hits.dwyl.com/Adamouization/Seal-Pup-Aerial-Imagery-Neural-Network-Classifier.svg)](http://hits.dwyl.com/Adamouization/Seal-Pup-Aerial-Imagery-Neural-Network-Classifier) [![GitHub license](https://img.shields.io/github/license/Adamouization/Seal-Pup-Aerial-Imagery-Neural-Network-Classifier)](https://github.com/Adamouization/Seal-Pup-Aerial-Imagery-Neural-Network-Classifier/blob/master/LICENSE)

This practical covers the exploration of various machine learning practices used for classifying different types of seal pups found in cropped aerial imagery obtained during seasonal surveys of islands. Various data visualisation techniques, data processing steps and classiﬁcation models are experimented with to design a final pipeline for making predictions on the types of seals observed in the images. The predictions are made by training multiple classiﬁcation models, which are all evaluated to determine their performance.

You can read the [full report here](https://github.com/Adamouization/Seal-Pup-Aerial-Imagery-Neural-Network-Classifier/blob/master/report/report.pdf).

## Dataset

The data comes in two ﬂavours: binary and multi, where:

* the "binary" data contains two labels, one for images of backgrounds and the other images of seals

* the "multi" data contains ﬁve labels, one for images of backgrounds and the four others for types of seals (whitecoat, moulted pup, dead pup and juvenile).

*Screenshot of the 5 different types of classification labels (top: HoG features, bottom: actual images):*
<p align="center">
  <img src="https://raw.githubusercontent.com/Adamouization/Seal-Pup-Aerial-Imagery-Classifier/master/report/figures/hog_multi.png?token=AEI7XLGCJXTHV3GI5RTJA2K62TRB4" alt="results" width="100%"/>
</p>

The dataset is not provided in this repository due to its large size. If you wish to experiment with the dataset on your own, please contact me and I will provide you a link to download the data.

## Results

#### Evaluating optimal neural network (3-fold cross validation)

<p align="center">
  <img src="https://i.postimg.cc/8CpZNjym/image.png" alt="training results" width="80%"/>
</p>

<p align="center">
  <img src="https://i.postimg.cc/KY3RK5sG/image.png" alt="training confusion matrix" width="85%"/>
</p>

#### Final testing results

These results are achieved on the testing dataset only once, unseen until this point:

* Binary: 98.21% accuracy

* Multi: 97.58% accuracy

## Usage

Create a new virtual environment and install the Python libraries used in the code by running the following command:

```
pip install -r requirements.txt
```

To run the program, move to the “src” directory and run the following command:

```
python3 main.py -s -d [-m ] [-gs] [-rs] [-v]
```

where:
* *"-s section"*: is a setting that executes different parts of the program. It must be one ofthe following: ‘datavis’, ‘train’ or ‘test’.
* *"-d dataset"*: selects the dataset to use. It must be set to either ‘binary’ or ‘multi’.
* *"-m model"*: is an optional setting that selects the classification model to use for training. It must be one of the following: ‘sgd’, ‘logistic’, ‘svc_lin’, ‘svc_poly’.
* *"-gs"* and *"-rs"*: are optional flags to run the hyperparameter tuning algorithms (eithergrid search or randomised search algorithms) for the selected classification model. The flag only takes effect when using multi-layer perceptron classifiers (neural networks).
* *"-v"*: is an optional flag that enters verbose (debugging) mode, printing additional statements on the command line.

## License 
* see [LICENSE](https://github.com/Adamouization/Seal-Pup-Aerial-Imagery-Neural-Network-Classifier/blob/master/LICENSE) file.

## Contact
* Email: adam@jaamour.com
* Website: www.adam.jaamour.com
* LinkedIn: [linkedin.com/in/adamjaamour](https://www.linkedin.com/in/adamjaamour/)
* Twitter: [@Adamouization](https://twitter.com/Adamouization)
