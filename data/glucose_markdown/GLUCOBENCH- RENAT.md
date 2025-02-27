# GlucoBench: Curated List of Continuous Glucose Monitoring Datasets with Prediction Benchmarks  

Renat Sergazinov1, Elizabeth Chun1, Valeriya Rogovchenko1, Nathaniel Fernandes2, Nicholas Kasman2, Irina Gaynanova1  

1Department of Statistics, 2Department of Electrical and Computer Engineering Texas A&M University  

![](images/60fe02757fcc577dc5d189ec9e68ada1bd85a4bba564957143a0bb3ab6dc9b61.jpg)  

March 25, 2024  

![](images/b2b5882d74e808d210a33fb44d06dc38e1a2599294ba547b33eb88401ff3b427.jpg)  

Figure: Sample of glucose curves captured by the Dexcom G4 Continuous Glucose Monitoring (CGM) system, with dates de-identified for privacy [5].  

Table: Summary of the glucose prediction models by dataset and model type.  

![](images/26a2f007267c4fdf492d58b868635b70524ac4b6b2c79f487b7e99049316cc5b.jpg)  

# Limitations:  

Lack of Benchmarks: few open datasets, no tasks, no pre-processing tools;   
Open-Source Shortage: 38 out of 45 surveyed methods released code;   
Narrow Focus: exclusion of Type 2 diabetes from the datasets.  

Table: Proposed suite of open datasets.   

![](images/b0122036c06df42a03cb69dadf733737c57dc087a0c17702bff2ff3003804ad8.jpg)  

# Data Tasks Empirical analysis References Our approach: pre-processing and tasks  

# Systematic pre-processing across datasets:  

Interpolation and Segmentation: linear interpolation or segment division.   
Covariates Scaling and Encoding: scaling and label encoding.   
Data Splitting: chronologically ordered $^+$ out-of-distribution set.  

To create a fair comparison and highlight main dififculties in CGM prediction,we create the following task setup:  

In-distribution fit: for patients in training data;   
Out-distribution fit: for new patients (cold start);   
Inclusion of covariates: support for covariates.  

![](images/65b9f96a02731b9cc7b455c1dce2024e9c9e90a752a870ab82955ed48f4bd5cc.jpg)  
Figure: Model forecasts on Weinstock [5] dataset.  

# In-distribution performance  

Table: In-distribution performance.   

![](images/98347dc7d8b7e07bada154d26d7ea97aecd2167175eaa01c60ffdea17f89ea54.jpg)  

# ask Empiric Out-of-distribution performance  

Table: with- vs. without-covariates, performance increase and decrease shown.  

![](images/2178778678793abe98a01fa52a1db937e01376fce99223ea7ce79e5fe9cf4f03.jpg)  

![](images/023ee5287e719f37bda37d8cedfec94d7f942ce8f257b50469fd82b1e52faaa0.jpg)  

Figure: Analysis of errors by: (a) OD versus ID, (b) population diabetic type (healthy $\rightarrow$ Type $_2\rightarrow$ Type 1), (c) daytime (9:00AM to 9:00PM) versus nighttime (9:00PM to 9:00AM).  

# Key takeaways:  

Model Performance Variation Factors:  

1 Dataset Size: deep learning models excel on larger datasets. 2 Patient Composition: healthy subjects being easier to predict than those with diabetes. 3 Time of Day: daytime predictions are more challenging. 2 Model Generalizability: 1 Deep learning models generally show better generalization. 2 Performance typically drops on out-of-distribution (OD) data. Impact of Covariates: 1 Integrating covariates is non-trivial, and currently no model is able to take full advantage of covariates.  

[1] S. Broll, J. Urbanek, D. Buchanan, E. Chun, J. Muschelli, N. M. Punjabi, and I. Gaynanova. Interpreting blood glucose data with r package iglu. PloS one, 16(4):e0248560, 2021.  

[2] A. Cola´s, L. Vigil, B. Vargas, D. Cuesta-Frau, and M. Varela. Detrended fluctuation analysis in the prediction of type 2 diabetes mellitus in patients at risk: Model optimization and comparison with other metrics. PloS one, 14(12):e0225817, 2019.  

[3] F. Dubosson, J.-E. Ranvier, S. Bromuri, J.-P. Calbimonte, J. Ruiz, and M. Schumacher. The open d1namo dataset: A multi-modal dataset for research on non-invasive type 1 diabetes management. Informatics in Medicine Unlocked, 13:92–100, 2018.  

[4] H. Hall, D. Perelman, A. Breschi, P. Limcaoco, R. Kellogg, T. McLaughlin, and M. Snyder. Glucotypes reveal new patterns of glucose dysregulation. PLoS biology, 16(7):e2005143, 2018.  

[5] R. S. Weinstock, S. N. DuBose, R. M. Bergenstal, N. S. Chaytor, C. Peterson, B. A. Olson, M. N. Munshi, A. J. Perrin, K. M. Miller, R. W. Beck, et al. Risk factors associated with severe hypoglycemia in older adults with type 1 diabetes. Diabetes Care, 39(4):603–610, 2016.  

# Thank You!  