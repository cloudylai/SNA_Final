# SNA_Final
Social Network Analysis Final Project

## Introduction
The repository is a duplication of the final team project, link prediction on location-based social network, of NTU Social Network Analysis and Graph Mining Course ([website](http://www.csie.ntu.edu.tw/~sdlin/Courses/SNA2014.html)) in Fall, 2014. The Project is worked by Chiu-Te Wang, Liang-Wei Chen, and Chih-Te Lai. The objective of this project is to predict different kinds of users' potential interest in some attractions. In this project, we build a heterogeneous network and extract complicate features from it. We train different models and testify our models on [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html) dataset. A completed report of details about the performance and architecture of our model is contained in results/ directory.

## Method
In order to perform link prediction, we build a supervised classifier based on random forest. Our main contribution is to design three kinds of features obtained by exploiting the topological structure of social network and behavior records among users. The three kinds of features are social/graph features (g), check-in/location features (c), and heterogeneous graph features (h). The detail of these three kinds of features are described in report. A diagram of our learning method is below. 

![image1]( https://github.com/cloudylai/SNA_Final/images/diagram1.png) 

## Result
The following figure is the AUC-ROC curve on the testing data. The blue curve stands for our baseline model which is trained by using one graphical feature. The green, the red, and the aqua curves represent the models which are trained by using (g), (g) + (c), (g) + (c) + (h) respectively. The result shows that our designed features improve the performance of learning classifier compared with baseline model.

![image2]( https://github.com/cloudylai/SNA_Final/results/exps_ROC2.png)



## Reference
There are many works related to location-based social network link prediction. In this project, we look into serveral works including:  
1. Scellato et al. [Exploiting Place Features in Link Prediction on Location-based Social Networks.](http://dl.acm.org/citation.cfm?id=2020575) in KDD 2011  
2. Chao et al. [Evaluating geo-social influence in location-based social networks.](http://dl.acm.org/citation.cfm?id=2398450) in CIKM 2012  
3. Mengshoel et al. [Will We Connect Again? Machine Learning for Link Prediction in Mobile Social Networks.](http://repository.cmu.edu/silicon_valley/152/) @ACM 2013  
4. Bayrak et al. [Contextual Feature Analysis to Improve Link Prediction for Location Based Social Networks](http://dl.acm.org/citation.cfm?id=2659499) in KDD 2014