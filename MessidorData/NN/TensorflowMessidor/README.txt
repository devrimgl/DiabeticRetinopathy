# Project Name

TODO: Write a project description

## Installation

TODO: Describe the installation process

## Usage

TODO: Write usage instructions

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