## 4. Now, consider the case where you had to manage a dataset with millions images
rather than a few hundred. How will you change your dataset building and storing
methods for:

* a. Faster access, given that data lives on the cloud infrastructure like S3

Sagemaker attaching to our data volumes will be fastest and likeliest most cost efficient with the availability of spot instances, especially when considering bandwidth cost

* b. Faster data re-sampling, to create custom datasets

A relational database management system would be ideal for this. Allowing us to curate very flexiable datasets for training or other types of analysis. A Schema will enable very powerful functionality.

* c. Faster data-loader access for faster training

In memory database such Redis or Memcache will dramatically reduce data fetch times
