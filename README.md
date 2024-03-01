# Supervised and unsupervised learning on Landsat image

Please read this carefully !!!!!!!!!!!!

There are three essential files.

"api.py" - This file logs into your USGS Earth Explorer account and downloads Landsat-9 images over a specified location. you need to provide coordinates and other parameters along with your USGS account username and password inside the Python file.
to run this file, go to your command prompt, then to your saved file directory, install all the dependencies, and then use the command "python3 api.py".

"unsupervised.py" - After downloading the images, The required band images are moved to another folder named "data1". this file takes all the seven bands as input and creates a 7-layer 2d-matrix stack. which is used as an input dataset for the KNN classifier. for detailed explanation check the code.
to run this file, go to your command prompt, then to your saved file directory, install all the dependencies, and then use the command "python3 unsupervised.py".

"supervise_svm_random_forest.py" - I have used a different dataset for supervised classification. The dataset folder "materials" along with shape files has been included in the folder. I have implemented a Support vector machine and random forest-based supervised classification. for detailed explanation check the code.
to run this file, go to your command prompt, then to your saved file directory, install all the dependencies, and then use the command "python3 supervise_svm_random_forest.py".

