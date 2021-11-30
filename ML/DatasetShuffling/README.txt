Here are a couple of notebooks used to shuffle the DeepSig 2018 dataset
One uses the dataset hosted on Kaggle, while the other uses google colab.
The reason for shuffling the dataset was to make splitting the dataset into training, validation,
and testing splits easier, and therefor better the training of ML models

Files included:
  - data-shuffling.ipynb
  - gc_kaggle_dataset_shuffling.ipynb
  - vgg_conv_shuffled.ipynb
  
 data-shuffling.ipynb:
    This notebook contains the first attempt to create a version of the DeepSig dataset that is shuffled. Was made 
    using Kaggle in mind.
    
 gc_kaggle_dataset_shuffling.ipynb:
    This notebook contains a working python script that creates a shuffled dataset. The dataset can be saved to 
    and hosted on google drive for future use
    
 vgg_conv_shuffled.ipynb:
     This notebook contains an ML model that uses the shuffled database. This was the first test using the new 
     dataset
