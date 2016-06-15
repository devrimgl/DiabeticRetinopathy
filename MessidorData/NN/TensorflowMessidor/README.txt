# Identifying Signs of Diabetic Retinopathy Using Deep Learning

Diabetic Retinopathy is a visual impairment disease as a result of dia-
betes mellitus.  Visual impairment caused by diabetic retinopathy(DR) can
be cured if detected in earlier stage of DR in patients.  In this work, we try
to automatically detect signs of DR by using deep neural networks, speci -
cally convolutional neural networks.  For this purpose, we train deep neural
network by using one of the widely used publicly available datasets and test
with other public datasets.  In the experiments we reach 80% accuracy and
0.9 area under curve results for some datasets.  Experimental results show
that  deep  learning  techniques  can  be  promising  in  the  task  of  automatic
detection of DR.

## Installation

PREPROCESS
---------------
For preprocessing the eye images, use convert.py. You have to change dataDirectoryPath variable in the settings.py file
for using different datasets.

TRAIN AND SAVE MODEL
====================
test_messidor_tf_twoClassEncode_Convolutional.py can be used to train and save a model to disk to be used by
test_with_other_datasets.py. Besides K-Fold cross validation experiments for DR NONDR experiments can be run
by using the same test_messidor_tf_twoClassEncode_Convolutional.py file. To use different datasets only paths settings.py
must be changed accordingly.

Similarly test_messidor_tf_oneClassEncode_Convolutional.py can be run for the experiments of degree of DR detection.
Notice that this time it is not binary classification.

For different configurations in the experiments, create_images_arrays() method must be modified accordingly. Contrast,
enhancement, image rotation and histogram equalization is used within this method. This method is in both
messidor_tf_twoClassEncode.py file and messidor_tf_oneClassEncode.py files.

## Usage

### Datasets

For this project, we have used following publicly available datasets. MESSIDOR is the main data set in this project.
MESSIDOR - ADCIS  CONSORTIUM.   Methods  to  evaluate  segmentation  and  indexing techniques  in  the   eld  of  retinal  opthalmology(messidor),  2009.    URL
http://www.adcis.net/en/Download-Third-Party/Messidor.html. [Online; accessed June 01, 2016].
STARE - Adam Hoover. Structured analysis of the retina, January 2013. URL http://
www.ces.clemson.edu/ahoover/stare/.  [Online; accessed June 01, 2016].
CFI - Balint  Antal  and  Andras  Hajdu.
Diabetic  retinopathy  debrecen  data
set,   2014a. URL   https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set.  [Online; accessed June 01, 2016].
HRF - Attila Budai. High-resolution fundus (hrf) image database, 2009. URL https:
//www5.cs.fau.de/research/data/fundus-images/.  [Online; accessed June
01, 2016].

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## History

TODO: Write history

## Credits

TODO: Write credits

## License

TODO: Write license

