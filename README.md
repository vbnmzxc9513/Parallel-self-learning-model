# Parallel-self-learning-model

This is a project to implement a machine learning system that can self-learning by collecting data from google using crawler. This system can be split into three major stages, the crawler stage, feature extraction stage, and SVM modeling stage. Each stage takes a long time, so we using lots of parallel methods to speed up them. Finally, we get approximately 85 times speedup over the serial program.
