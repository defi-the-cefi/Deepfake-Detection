## 3. Write a simple (supervised) deep classifier to train and test using the dataset
collected in Q1.

* a. How will you divide your dataset into training and test sets.

70/100 train, 15/100 validation, 15/100 test, with 50/50 fake, real images in each 

* b. What data-augmentation techniques will to use for out-of-distribution (unseen) images?

Random crop, Random Flip, Random Invert, Random Rotate, Random Sharpness Adjust, Random Contrast Adjust

* c. Please test accuracy on the attached, rd_test_dataset zipped face images, and save the output to a .csv file.
https://drive.google.com/file/d/1jcdByJPkAGq9JsgsdLqeyLwI4Yl6plOf/view?usp=sharing


Annotations would not load on a non-MAC OS, but the code to do inference on the RD test set can be found [here](https://github.com/defi-the-cefi/RD-Data-Takehome-Anthony-Zelaya/blob/e7e07463ee99a309f3164b69ec9d60ae89a55ded/deepfake_detection/efficient_net_b7.py#L86)

* d. Explain your accuracy scores, and add analysis based on what you proposal in
Q2.

Need more data and training time. Would help a lot to actually train on videos as most generative models still seem to struggle with temporal consistency and that might be the best place to get bang for your buck in fakespotting. Would also benefit from training on files of varying compression as artifacting introduces noise that may throw off detection.
