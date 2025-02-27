# Convolutional Recurrent Neural Networks for Glucose Prediction  

Kezhi Li, John Daniels, Chengyuan Liu, Pau Herrero, Pantelis Georgiou Department of Electronic and Electrical Engineering, Imperial College London, London SW7 2AZ, UK  

Abstract—Control of blood glucose is essential for diabetes management. Current digital therapeutic approaches for subjects with Type 1 diabetes mellitus (T1DM) such as the artificial pancreas and insulin bolus calculators leverage machine learning techniques for predicting subcutaneous glucose for improved control. Deep learning has recently been applied in healthcare and medical research to achieve state-of-the-art results in a range of tasks including disease diagnosis, and patient state prediction among others. In this work, we present a deep learning model that is capable of predicting glucose levels over a 30-minute horizon with leading accuracy for simulated patient cases (RMSE $\mathbf{\Sigma=9.38{\pm0.71}\:[m g/d L]}$ and $\mathbf{MARD}=5.50{\pm}0.62\%)$ and real patient cases $(\mathbf{RMSE}=21.13{\pm}1.23\;[\mathbf{mg}/\mathbf{dL}]$ and $\mathbf{MARD}=\mathbf{10.08}{\pm}\mathbf{\pm0.83}\%)$ . In addition, the model also provides competitive performance in forecasting adverse glycaemic events with minimal time lag both in a simulated patient dataset $(\mathbf{MCC}_{h y p e r}=\mathbf{0.83{\pm0.05}}$ and $\mathrm{~\breve~{~\Rightarrow~}~}\mathbf{MCC}_{h y p o}=\mathbf{0.80\pm0.10})$ and in a real patient dataset $(\mathbf{MCC}_{h y p e r}$ $\mathbf{\Sigma=\0.79\pm0.04}$ and $\mathbf{MCC}_{h y p o}\ =\ \mathbf{0.38{\pm}0.10})$ . This approach is evaluated on a dataset of 10 simulated cases generated from the UVa/Padova simulator and a clinical dataset of 5 real cases each containing glucose readings, insulin bolus, and meal (carbohydrate) data. Performance of the recurrent convolutional neural network is benchmarked against four algorithms. The prediction algorithm is implemented on an Android mobile phone, with an execution time of 6ms on a phone compared to an execution time of 780ms on a laptop in Python.  

Index Terms—Type 1 diabetes, continuous glucose monitor (CGM), glucose prediction, deep learning, long short term memory (LSTM).  

# I. INTRODUCTION  

Diabetes is a chronic illness characterised by a lack of glucose homeostasis. A healthy pancreas dynamically controls the release of insulin and glucagon hormones through the $\alpha$ - cells and $\beta$ -cells respectively, in order to maintain euglycaemia [1]–[3]. In Type 1 diabetes, an autoimmune disease, the $\beta$ - cells are compromised and therefore suffer from impaired production of insulin. This leads to increased periods of hyperglycaemia (blood glucose concentration $>180m g/d L)$ and hypoglycaemia (blood glucose concentration $<\ 70m g/d L)$ [4], [5]. Insulin therapy is needed to manage blood glucose (BG) levels to maximise the time in (advised) target [2], [6].  

The standard approach to diabetes management requires the subject to actively undertake blood glucose measurements a handful of times throughout the day with a finger prick test - self monitoring of blood glucose (SMBG). The recent development and uptake of continuous glucose monitoring (CGM) devices allow for improved sampling (5 minutes) of glucose measurements. This approach has proven to be effective in enabling glucose control and thus improving the outcome of subjects in clinical trials relative to standard approaches [6]–[8]. Further improvement of glucose control can be realised through prediction, allowing users to take action ahead of time to minimise the occurrence of adverse glycaemic events. The challenge lies in the multiple factors such as insulin variability, ingested meals, stress and other physical activities that influence glucose variability [9]. In addition, the glycaemic responses of different people to the same conditions can be very different, which is conditioned by high subject variability.  

Machine learning covers a range of techniques that allow intelligent systems to tackle problems by learning and extracting patterns in data to build appropriate models. Consequently, the techniques involve discovering a suitable mapping from the representation of input data to the output. The performance of traditional machine learning algorithms such as logistic regression, $\mathbf{k}$ -nearest neighbours [10], or support vector regression [11] heavily depend on the representation of the data they are given. Typically, the features - information the representation comprises - are engineered with prior knowledge and statistical features (mean, variance) [12], principal component analysis (PCA) [13] or linear discriminant analysis [14]. Artificial neural networks are also investigated widely in diabetes management [15]–[19]. However, convolutional neural networks are mostly used with less than 3 layers, which limits the power of neural networks with current training techniques.  

Deep learning offers an approach to automatically learn both the features and consequent nonlinear mapping to the output. This approach, which incorporates multi-layer neural networks has been effective in solutions that have improved on previous upper bounds of performance in computer vision [20], board games [21], diseases diagnosis [22], [23], and healthcare [24]–[26]. Deep learning typically shows superior performance to traditional techniques due to this ability to automatically learn features with physical meanings [27], [28], hence potentially encoding features that may not be previously known to researchers.  

In this paper, we propose a deep learning algorithm for glucose prediction using a multi-layer convolutional recurrent neural network (CRNN) architecture. The model is primarily trained on data comprising CGM, carbohydrate and insulin data. After preprocessing, the time-aligned multi-dimensional time series data of BG, carbohydrate and insulin (other factors are optional) are fed to the deep neural network (DNN) for training. The architecture of the DNN is composed of three parts: a multi-layer convolutional neural network that extracts the data features using convolution and pooling, followed by a recurrent neural network (RNN) layer with long short term memory (LSTM) cells and fully-connected layers. In the LSTM cell, an input gate, an output gate and a forget gate are incorporated to remember the previous model state and update the model state simultaneously when current data are fed seamlessly. The convolutional layer comprises a 1D Gaussian kernel filter to perform the temporal convolution, and pooling layers are used for reducing the feature set. LSTM are the variant of gated RNN used since it shows good performance in predicting time series given long time dependencies [29]. In addition, it deals well with the vanishing gradient problem for training traditional RNNs. The final output is a regression output by the fully connected layer. The DNN is realized using the open-source software library Tensorflow [30], and it can be easily implemented to portable or wearable devices with its simplified version Tensorflow Lite. The performance of the proposed method is evaluated on datasets of simulated cases as well as clinical cases of T1DM subjects, and compared against the latest algorithms including support vector regression (SVR) [11], the latent variable model (LVX) [31], the autoregressive model (ARX) [32], and neural network for predicting glucose algorithm (NNPG) [15], which show the competitive performance of the method.  

The paper is organized as follows. Section II briefly discusses the setup and specification of the system, the pipeline for preprocessing the data. Section III details the architecture of the proposed multi-layer convolutional recurrent neural network. The performance of the proposed method is evaluated and discussed in Section IV. Finally, Section V concludes the paper with a discussion of challenges in this approach and avenues for future research.  

# II. METHODS  

After introducing the data acquisition and hardware platform, the method of the glucose prediction is explained. The approach consists of several components: preprocessing, feature extraction using CNN, time series prediction using LSTM and signal converter. The architecture of the proposed CRNN is shown in Fig. 2. In the diagram, the input of the algorithm is time series of glycaemic data from CGM, carbohydrate and insulin information (time and amount); other related information are optional (exercises, alcohol, fat, stress, etc.). The output of preprocessing is cleaned as time-aligned glycaemic, carbohydrate and insulin data, which are then fed to the CNN. The output of CNN serves as the input of RNN, which is a multi-dimensional time series data, representing the concatenation of features of the original signals. The output of the RNN is the predictive blood glucose level 30-min later, while hidden states are inherited and updated continuously internally inside of the RNN component. We evaluate the models with 30-min prediction horizon (PH) because it is widely used in glucose prediction software, and is easier to compare results with other works [11], [15], [17], [31], [33], [34]. We proceed with an explanation data pipeline and components of the model architecture.  

![](images/1eee6bcd5c55d4c573705073b5bd6027569b69a25ecf62f2cf0c3c2c68fdbb0a.jpg)  
Fig. 1. A brief flow diagram explains the procedure to implement deep learning models to portable devices with the help of TF Lite. In the figure, the yellow square frames denote the models or files obtained after each operation, and the yellow round frames denote the associated operations applied. The file can be installed in Android or iOS app, with slightly different settings.  

# A. Data Acquisition and Implementation  

The data used in this paper include two datasets, simulated data and clinical data. Simulated data of T1DM subjects was generated using the UVA/Padova T1D, which is the only simulator for glucose level simulation approved by the Food and Drug Administration (FDA). Clinical data was obtained from a 6-month clinical dataset corresponding to 10 adult subjects with T1D undergoing a clinical trial evaluating the benefits of an advanced insulin bolus calculator was employed [35]. The information included in the dataset needs glucose, meal, insulin and associated time stamps. Other information is optional for the method.  

After the model has successfully undergone training and validation, we also implement our algorithm on mobile phones through Tensorflow Lite due to its efficiency running on portable devices. The model is converted to a Lite model file and installed on an Android or iOS system. It needs the associated application programming interface (API) and interpreter to carry out the inference. Fig. 1 shows how the Tensorflow Lite model file [30] is wrapped and loaded in a mobile-friendly format.  

# B. Outlier Detection and Filtering  

In the preprocessing component, the main purpose is to clean the data, filter the unusual points and make it suitable as the input to the CNN. Besides the normal steps including time stamp alignment, normalization and missing data imputation for time series data with trend, the most important operation to improve the data quality is the outlier detection, interpolation and filtering, in particular for clinic data. Because in the clinical data of glucose time series, there are many missing or outlier data points due to errors in calibration, measurements, and/or mistakes in the process of data collection and transmission. Here, several methods are introduced to handle these scenarios [36].  

• Dimension Reduction Model: the time series can be projected into lower dimensions using linear correlations such as principle component analysis (PCA) [37], and  

![](images/363cd04b149ba5e413d5a52cb9a9f500421534c3f6c112a3351b5739b1ba5778.jpg)  
Fig. 2. The architecture of the proposed convolutional recurrent neural network for blood glucose prediction. The data at the left is the concatenated time series data including glucose level, carbohydrate, insulin and other factors. After outlier filtering, the multi-dimensional data can be sent to the multi-layer convolutional component. Then the resultant time series is sent to the modified recurrent neural network component presented in a red frame, which includes LSTM cells and dense fully connected layer. Finally, the resultant is converted back from “change of the glucose value” to “absolute glucose value”. The output is the future glucose values of PH (eg. $\mathrm{PH}=30\$ mins).  

data with large residual errors can be considered as outliers.   
Proximity-based Model: the data are determined by nearest neighbour analysis [38], cluster or density [39]. Thus the data instances that are isolated from the majority are considered as outliers.   
• Probabilistic Stochastic Filters: different filters for the signals, such as gaussian mixture models optimized using expectation-maximization. In [40] a similar gaussian filter is implemented as a kernel to filter the 1D signal in the CNN. While in our case the filter can be implemented before the CNN, due to the continuous characteristic of the input glycaemic time series data.  

For some cases when the data fluctuates with high frequency, 1D Gaussian kernel filter is implemented on the glucose time series to get rid of the noise. The result of this process is a smoothed, continuous time series of glycaemic data from the CGM, as well as time-aligned carbohydrate and insulin information. We feed the model sample points that from this 3-dimensional data that cover the last 2 hours before the current time in the time series. A sliding window of size 24 is therefore determined to train the model, because we found it is an appropariate setting to balance the tradeoff between the prediction accuracy and the computation complexity.  

# C. A Multi-layer Convolutional Network  

The filtered time series signal goes through the multilayer convolutions, which transforms the input data into a set of feature vectors. The convolution operation follows the temporal convolution definition in which  

$$
z[m]=\sum_{i=-l}^{l}x[i]\cdot\delta[m-i],
$$  

where $x$ represents the input signal, $\delta$ denotes the kernel, and $z$ is the result of the convolution. The input signal can be fed block by block, using a sliding window setting. The windows can be overlapped or non-overlapped, determined by the allowed CNN size and computations. The CNN automatically learns the associated weights and recognises particular patterns and features in the input signal that can best represent the data for future time steps, using the back-propagation technique.  

The proposed CNN has 2 hidden layers, consisting 2 convolutional layers, with max pooling applied to down-sample the feature map obtained from the previous convolutional layer. It is common to periodically insert a pooling layer in-between successive convolutional layers to progressively reduce the size of the representation, thus the amount of parameters as well as the computation in the CNN. It also guards against the problem of overfitting. For instance, if the accepted size is $L1\times D1$ , and the down-sampled parameters are spatial extent $F$ and stride $S$ , and it results in a max-pooling vector $Y$ of size $L2\times D2$ where  

$$
\begin{array}{l}{L2=(L1-F)/S+1}\\ {D2=D1}\\ {Y_{i}=\operatorname*{max}(y_{i}^{*})}\end{array}
$$  

where $y_{i}^{*}$ is the vectors after down-sampled, $Y_{i}$ is the feature map and $\operatorname*{max}()$ is the operator that computes the maximum value. During the training process, the CNN is trained by backpropagation and the stochastic gradient descent method. The initial weights of the network are set randomly, and the meanabsolute-error is set as the cost function to be minimised in the training. Partial derivatives of the error in terms of the weights $w_{i}$ and bias $b_{i}$ are computed, and the associated $w_{i}$ and $b_{i}$ are updated accordingly. The last convolutional layer then feeds directly into the recurrent layer that makes up the next sub-component in the architecture.  

![](images/96f13ac37b0aef514976b6863b592d077f8e155cc7fe38e592487ecc8dcf8e49.jpg)  
Fig. 3. An illustration of the proposed modified recurrent neural network in the inference. It origins from the conventional LSTM cell, including an input gate, an output gate and a forget gate. The difference lies in the output part, which indicates in red colour in the figure. The output of $h_{t}$ is still the internal parameter which is transmitted to the next cell. However, the output value signal is $y_{t}$ instead of $h_{t}$ . It is because in the training process, the target of RNN is the change of the glucose value between current time and 30 mins later. Thus the predictive glucose level after 30 mins should be the inference value plus the current glucose value.  

![](images/4a467dbb860c4cac6da87d674c866a5949f38e06c4d81a19be167d1056cb8043.jpg)  
Fig. 4. An illustration of the proposed CNN architecture. The multidimensional aligned time series data are concatenated, and then sent to a multi-layer CNN composed of convolutional layers and pooling layers. Finally, after going through a fully connected layer, the final output is the summation of the model output and a copy of the original CGM time series.  

# D. A Modified Recurrent Layer  

The RNN implemented in our model is a LSTM network, which is composed of 64 LSTM cells [41]. Each LSTM cell consists of an input gate, an output gate and a forget gate. Each of the three gates can be thought of as a neuron, and each gate achieves a particular function in the cell. The LSTM network is good at building predictive models for time series and sequential data [42]. This is because the cell remembers the previous data patterns over arbitrary time intervals, thus the internal “memory” can predict the future output according to the previous experience. Its memory is updated when new data are fed to the model seamlessly.  

The output of the CNN, a 11-dimensional time series, is used to train the LSTM network. We generated a RNN as 1 hidden layer, consisting of a wide LSTM layer. The LSTM layer has 64 cells, with a dropout applied in this layer. Dropout refers to ignoring neurons randomly during the training phase of certain set of neurons. In many cases it has been verified that dropout can effectively avoid the overfitting problem and improve genralisation [43], [44].  

The main difference between a normal LSTM and the proposed LSTM is that the proposed model has the transform and recover steps to modify the BG values before going through and after coming out the conventional LSTM model. Because in training, instead of using the BG values directly, we use the change of BG between the current BG $x(t)$ and the future BG $x(t\!+\!6)$ as the target label. It is called the transform step. In the training, the input sliding window matrices are multi-dimensional time series (including BG values, meal, insulin). After the model has been trained, the inference output is the change of BG $\triangle x(t)$ between $x(t)$ and $x(t+6)$ . So we need the BG value $x(t)$ to obtain the prediction of BG at time $t+6$ with $x(t+6)=x(t)+\triangle x(t)$ . It is called the recover step.  

Specifically, a modified LSTM RNN estimates the conditional probability $p(y_{1},\cdot\cdot\cdot\ ,y_{T^{\prime}}\vert x_{1},\cdot\cdot\cdot\ ,x_{T})$ given a sequence of data, where $x_{1},\cdot\cdot\cdot,x_{T}$ denotes the input sequence and $y_{1},\cdot\cdot\cdot,y_{T^{\prime}}$ is the corresponding output sequence, with size $T$ and $T^{\prime}$ respectively. If $x_{t},h_{t},c_{t}$ are used to denote the input vector, output vector and memory cell vector respectively; $W,U.$ , and $b$ are the parameter matrices/vectors that can be learned in the network. $f_{t},i_{t}$ , and $o_{t}$ denote the forget gate, input gate, and output gate vectors. Then mathematical form of the update process can be explicitly written as  

$$
\begin{array}{r l}&{f_{t}=\sigma_{g}(W_{f}x_{t}+U_{f}h_{t-1}+b_{f})\ \ i_{t}=\sigma_{g}(W_{i}x_{t}+U_{i}h_{t-1}+b_{i})}\\ &{o_{t}=\sigma_{g}(W_{o}x_{t}+U_{o}h_{t-1}+b_{o})\ \ g_{t}=\sigma_{t}(W_{g}x_{t}+U_{g}h_{t-1}+b_{g})}\\ &{c_{t}=f_{t}\circ c_{t-1}+i_{t}\circ\sigma_{t}(g_{t})}&{h_{t}=o_{t}\circ\sigma_{t}(c_{t}),}\\ &{y_{t}=o_{t}\circ\sigma_{t}(c_{t})+x_{t},}\end{array}
$$  

where $\sigma_{g},\sigma_{t},\circ$ is the sigmoid function, hyperbolic tangent and entrywise product, respectively. In (3), the 1st to 5th equations are the same to the equations of normal LSTM. However the last 2 equations of $h_{t}$ and $y_{t}$ are modified according to the proposed glucose model. It has been shown in Figure 3. In the figure, the difference between the proposed modified LSTM and the normal LSTM is indicated using the red colour. The output of the LSTM cell is not $h_{t}$ , but $y_{t}$ . The $h_{t}$ is used as a remembered state that goes to the next time stage. However the output $y_{t}$ is calculated from $h_{t}$ plus the original input $x_{t}$ .  

That is because the output (and targets) of the neural network is the change of the blood glucose level. The real value of the predictive glucose level needs to be recovered from the glucose change by adding the baseline glucose value.  

Finally, the last layer of RNN feeds a multi-layer fully connected network, which consists of 2 hidden layers (256 neurons and 32 neurons) and an output layer (a single neuron) with the glucose change as output. The fully connected layer produces the output with an activation function  

$$
Z_{i}=a c t(\sum_{i=1}^{N}Y_{i}w_{i}+b_{i}),
$$  

where $Z_{i}$ is the multi-dimensional output, $a c t()$ is an activation function, $w_{i}$ and $b_{i}$ are weights of the fully connected network that can be trained. Particularly, $a c t()$ can be chosen from a set of activation functions such as sigmoid function $a c t(a)\:=\:1/(1+e^{-a})$ , rectifier $a c t(a)\,=\,\log\left(1+\exp(a)\right)$ or simple linearly $a c t(a)\,=\,a$ . In this paper we choose the simple liner function $a c t(a)=a$ as the activation function for its simplicity.  

In the training, mean-absolute-error between the target and the predictive value are being minimised. The optimiser we use is RMSprop, because it is usually a good choice for recurrent neural networks. It maintains a moving (discounted) average of the square of gradients, and divides gradient by the root of this average.  

# III. RESULTS  

In this section we test the proposed CRNN algorithm for glucose level prediction using both in silico dataset and a clinical dataset, and compare with state-of-the-art algorithms developed in recent years.  

# A. Verification Set-up  

For all methods, the datasets include CGM data recorded every 5 minutes, meal data indicating meal time and amount of carbohydrates, as well as insulin data with each bolus quantity and the associated time. Exercise, stress and alcohol consumption were included in the datasets but as optional variables in the model. In the data preprocessing step, incidences of meal and insulin intake are aligned to CGM data with the same timestamps.  

The in silico data considered in this section is generated via UVA/Padova T1D [45]. This simulator is approved by the FDA and therefore, serves as robust and validated framework for generating simulated cases. The cohort of T1D cases generated can be configured with varying meal and insulin information such that each case sufficiently differs. We generate a dataset of 10 unique adult cases and each has 360 days of data for each case. There are 3 meals per day. Insulin entries vary in each day, from 1 to 5. The insulin entry can be with a meal (meal and insulin at almost the same time), or without a meal (correction bolus). Exercise is considered at certain points in the data as well, which occur at any time except for nights. The intra-subject variabilities are also introduced for the simulator. The training data accounts for $50\%$ of the dataset, and the testing set is the rest of the data.  

The clinical data was collected from T1DM subjects in a 5 month clinical trial. The CGM data was measured using Dexcom G4 Platinum CGM sensors , with measurements received every 5 minutes. The CGM sensors were inserted from the first day of the study, and calibrated according to the manufacturer instructions. Other information available in the dataset such as meal, insulin, exercises was logged by the diabetic subjects themselves. Though the data has good quality, there still exist many periods of missing data, bad points or unexpected fluctuations. The diabetic subjects behaved normally during the study as they would in their daily lives. Similar to the in silico experiment, each subject’s clinical data is halved for training and testing data.  

# B. Criteria for Assessment  

Several criteria are used to quantitatively assess the performance of the proposed prediction algorithm. The root-meansquare error (RMSE) and mean absolute relative difference (MARD) between the predicted and reference glucose readings - for simulated or clinical CGM data - serve as the primary indicator to evaluate the general predictive performance of the algorithm.  

$$
R M S E=\sqrt{\frac{1}{N}\sum_{k=1}^{N}(y(k)-\hat{y}(k|k-P H))^{2}},
$$  

where $\hat{y}(k|k-P H)$ denotes the prediction results provided the historical data and $y$ denotes the reference glucose measurement, and $N$ refers to the data size.  

$$
M A R D=\frac{1}{N}\sum_{k=1}^{N}\frac{\lvert\hat{y}_{k}(k\lvert k-P H)-y(k)\rvert}{y(k)},
$$  

The RMSE and MARD provide an overall indication of the predictive performance of the proposed algorithm. As mentioned earlier, the benefit of glucose prediction is in avoiding adverse glycaemic events. In this clinical context, these metrics are limited in the insight they provide. Additional metrics are needed to assess the proposed algorithm in the following perspective:  

Capability of the forecasting algorithm in differentiating between adverse glycaemic events and nonadverse glycaemic events.   
Time delay in the predicted glucose readings and the reference values to evaluate the response time provided to deal with the potential adverse glycaemic event.  

The Matthews Correlation Coefficient (MCC) is used to evaluate the performance of the algorithms for detecting either adverse glycaemic event (hypoglycaemia or hyperglycaemia).  

$$
M C C=\frac{(T P\times T N)-(F P\times F N)}{\sqrt{(T P+F P)(T P+F N)(T N+F P)(T N+F N)}},
$$  

where $T P,F P,F N,T N$ stand for true positive, false positive, false negative, and true negative respectively. Here, positive indicates a hypoglycaemia $(<\ \ 70\ \ \mathrm{mg/dL})/$ hyperglycaemia $(>\ \ 180\ \ \mathrm{mg/dL})$ event in the next 30 minutes, and true means that the classification is correct. A standard confusion matrix typically includes the Accuracy as opposed to Matthews Correlation Coefficient (MCC). This modification addresses the imbalance in classes inherent in this situation - non-adverse events far outweigh adverse events.  

TABLE IRMSE AND MARD COMPARISON OF DIFFERENT PREDICTION METHODS FOR 10 VIRTUAL ADULT DIABETIC SUBJECTS  

![](images/6a41d3be54dedc59ab9eca6fc03ec7c5820e2f736d0530f96cfdfc6d8f295621.jpg)  
$p$ -value $\leq\,0.05$ $^{\dagger}p$ -value $\leq0.01$ $^{\ddagger}p$ -value $\leq0.005$  

TABLE IICOMPARISON OF ACCURACY MEASURES IN DIFFERENT PREDICTION METHODS FOR 10 VIRTUAL ADULT DIABETIC SUBJECTS  

![](images/25e3ea99702da2065883587ad763fd4609c749c3c69d5e6456e01269f5d52f41.jpg)  
\* $p$ -value $\leq\,0.05$ $\overline{{\uparrow}}_{p}$ -value $\overline{{\leq0.01\ ^{\ddagger}p}}$ -value $\leq\,0.005$  

The effective prediction horizon is defined as the prediction horizon, taking into account delays due to the responsiveness of the algorithm for a predicted value relative to its reference value. Cross correlation of the predicted and actual readings is employed in performing a time delay analysis of the proposed algorithm to determine the effective prediction horizon.  

$$
\begin{array}{r}{P H_{e f f}=P H-\tau_{d e l a y}\qquad\qquad\qquad\qquad\quad}\\ {=P H-\arg\operatorname*{max}(\hat{y}_{k}(k|k-P H)\star y(k))}\end{array}
$$  

A singular quantitative metric is not sufficient in evaluating performance of the proposed algorithm. Consequently, the set of metrics collectively give a comprehensive description of the quality of the prediction algorithm performance.  

# C. In Silico Data  

The performance of the proposed algorithm is contrasted with that of four baseline methods: NNPG, SVR, LVX and ARX (3rd order). The results are compared after the same data pre-processing. The performance of the algorithms are compared based on the accuracy over a 30-minute prediction horizon. Different algorithms were tested on the in silico data generated in a way described previously. The parameters involved in these algorithms are tuned carefully for optimal result. In SVR, the SVR function in Python is applied with the optimal parameters $(C=1e2,\gamma=0.01,c a c h e_{s i z e}=1000)$ . The LVX method are applied based on the MATLAB code provided in [31], the optimal predictor length and the number of LVs are $J_{x}=4$ and $N_{L V}=4$ respectively - this represents 20-minute historical data of glucose measurement, insulin and meal information being used for prediction. The 3rd order ARX model is optimized by MATLAB function arx() for every specific subject. The results of RMSE and MARD are summarized in the Table I and the results of forecasting of adverse glyvaemic events are summarized in the Table II.  

In Table I, we compare the predictive error of the algorithms to measure the accuracy of the algorithms. The CRNN algorithm exhibits the best overall RMSE and MARD for the 10 simulated cases. The results in Table I are statistically significant relative to each algorithm. This observation is also evident in both the hyperglycaemia and hypoglycaemia region. In the hyperglycaemia region the CRNN shows a statistically significant improvement in glucose prediction in adverse regions (hyperglycaemia and hypoglycaemia) over other algorithms, with the exception of LVX where the improvements in RMSE $(+3.13\mathrm{mg/dL}$ and $+0.28\mathrm{mg/dL})$ and MARD( $+0.62\%$ and $+0.27\%)$ are not statistically significant. However, Table II shows the CRNN report a statistically siginificant improvement in effective prediction time $(+1.5\mathrm{min})$ over LVX. The CRNN model also reports relatively low standard deviations from which we infer a benefit in building individualized models.  

An illustration of a comparison of various algorithms are shown in Fig. 5 for a virtual adult 4. As seen in the Figure 5, the CRNN exhibits the best responsiveness as the predictive curve responds rapidly towards the sharp glycaemic uptrend. The algorithm learns representations that appropriately account for both sharp slopes and gradual increments in the glycaemic curve. Consequently, at a glycaemic peak, CRNN yields a predictive curve with even higher slope to compensate the time lag aiming at reducing the gap between the prediction and real measurements. This feature helps CRNN to decrease the RMSE and MARD as well as maximising the effective prediction horizon, as shown in Figure 5.  

![](images/040d1f5a0e810d0e42448d71ecae897053674f4e159a4239e80184f07369258c.jpg)  
Fig. 5. One-day period prediction results for virtual adult 4. The solid black line, dotted green line, solid magenta line, dashed blue line, dash-dotted red line indicate the simulated glucose measurements, the prediction results of the 3rd order ARX method, the prediction results of the SVR method, the prediction results of the LVX algorithm, the prediction results of the CRNN method, respectively.  

# D. Clinical Data  

As mentioned in the previous section, the data obtained in the clinical trial exhibits missing data, and erroneous data. This results in non-physiological discontinuities that would affect the training process. To mitigate these occurrences, the data is processed with interpolations for gaps in data, yet the interpolation points are not included in all calculations.  

Table III shows the RMSE and MARD of the performance of the algorithms for the 5 cases of real data. The CRNN still maintains the best results for both the RMSE and MARD relative to the baseline methods. Relative to the performance achieved in the simulated datasets, the performance with real clinical shows worse average RMSE and MARD. This can be attributed to added external factors leading to discontinuity in data - more variability and unaccounted perturbations.  

Table IV shows that the CRNN maintains the best performance in regard to detection of hyperglycaemic events. Given results in the previous section, it was observed that the RMSE and MARD of the various algorithms pointed to reduced performance between simulated cases and real cases. This observation is also made in regard to the performance in accuracy in detecting adverse glycaemic events. The reduction in performance in detecting hypoglycaemia is not significant statistically relative to the performance exhibited in other algorithms. Regarding the effective prediction horizon, CRNN maintains a better effective prediction horizon $(+7.2\ \mathrm{\min})$ than LVX which would give a potential subject more time to avert an adverse glycaemic event. It would be more useful due to the effective prediction horizon, and CRNN has a better hypoglycaemic and hyperglycaemic accuracy in terms of RMSE. This is a significant improvement over the other baseline methods.  

As seen in Figure. 6, the CRNN and LVX both achieve good predictive curves compared to the ground truth measurements. Specifically, at the inflection periods during peaks and troughs, the LVX tends to have higher and lower predictions, respectively.The CRNN follows the trend at both local and global peak points closely, which increases its overall accuracy.  

# IV. DISCUSSION  

# A. Performance in Simulated and Clinical Data  

In the in silicon dataset, more subjects with T1DM and more complex scenarios (such as food, insulin, exercises) can be simulated. While for the clinical dataset, as real data collected from clinical trials, they are more practical and significant if people want to compare the performances of different methods. As a result, both in silicon and clinical dataset are used in the results section.  

In the previous section we noted a discrepancy in the performance of the proposed algorithm and baseline algorithms in simulated cases and the real patient cases. In our opinion, the drop in performance can be primarily attributed to the increased complexity of real data generated from a patient relative to the simulated data generated from a physiological model. In addition the gaps in data and method of interpolation may contribute to the further reduction in performance of the model. Relative to the baseline algorithms, the CRNN is better at capturing the features since deep learning affords a better capacity at learning optimal representations of features. This would explain the relatively lower variance in metrics for the performance of the CRNN in different cases relative to baseline models. Finally, it is also noted in the hypoglycaemia event forecast where the LVX exhibits marginally better - though statistically insignificant - result over CRNN $(+0.01)$ . This could be attributed to the reduced number of hypolgycaemia cases present in clinical data to train on relative to simulated cases where CRNN exhibits best results in hypoglycaemia forecasting.  

TABLE III RMSE AND MARD COMPARISON OF DIFFERENT PREDICTION METHODS FOR 5 REAL ADULT DIABETIC SUBJECTS   

![](images/1af396096a7a3a014636efb44f406e920e96e0704a4c3d471c25e6f32f9f7450.jpg)  
\* $p$ -value $\leq\,0.05$ $\overline{{\uparrow}}_{p}$ -value $\le0.01$ $\overline{{\ddag_{p}}}$ -value $\leq0.005$  

TABLE IV COMPARISON OF ACCURACY MEASURES IN DIFFERENT PREDICTION METHODS FOR 5 REAL ADULT DIABETIC SUBJECTS.   

![](images/76ef7af67b8bc75cd3c73d104293eccdd2b589e11b1d0073cd637ace7e524a3b.jpg)  
$p$ -value $\leq\,0.05$ $\top_{p}$ -value $\leq0.01$ $\overline{{\ddag_{p}}}$ -value $\leq0.005$  

![](images/d0fa2b732fa61b6f33a52ff44a19427e462b5fc58cc60682888e0decf5f95aa8.jpg)  
Fig. 6. One-day period prediction results for clinical adult 17. The solid black line, dotted green line, solid magenta line, dashed blue line, dash-dotted red line indicate the simulated glucose measurements, the prediction results of the ARX method, the prediction results of the SVR method, the prediction results of the LVX algorithm, the prediction results of the CRNN method, respectively.  

# B. Results Comparisons  

We achieved a mean $\mathrm{RMSE}=9.38\mathrm{mg/dL}$ in silico using the proposed method, and it is the best amongst other advanced algorithms, including SVR, LVX and 3rd order ARX. In addition, we want to compare our algorithm with other approaches in the literature. Also working on glucose dataset generated from simulators, our algorithm is also better than the results of $\mathrm{RMSE}\,=\,18.78\mathrm{mg/dL}$ [33] and $\mathrm{RMSE}\,=\,13.65\mathrm{mg/dL}$ in [34].  

For several other works, it is difficult to evaluate the RMSE through direct comparison due to the unavailability of code and parameters for replicating these models. In addition, the lack of a benchmark dataset means the performances are of computed on different datasets. However, we may compare the results with widely used methods as benchmarks, such as SVR or NNPG. For instance, for $\mathrm{PH}=30~\mathrm{min}$ as shown in Table 3 [17], the algorithm is $0.1\ \mathrm{mg/dL}$ better than the result of SVR in terms of RMSE on the real dataset; our algorithm is $6.0~\mathrm{mg/dL}$ better than the SVR in terms of RMSE on the real dataset. In [16], for $\mathrm{PH}=30$ min their RMSEs are 1.3 better than NNPG for the simulated data and $0.2~\mathrm{mg/dL}$ better than NNPG for the real datasets. Our RMSEs are $3.1~\mathrm{mg/dL}$ better than NNPG for the simulated data and $6.1\ \mathrm{mg/dL}$ better for the real datasets. As far as we are aware, the proposed algorithm achieves a performance state-of-the-art accuracy with regard to RMSE.  

# C. Parameters in ML and Hyperparameters in DL  

For conventional machine learning tasks, feature selection is crucial to achieving a good performance. The quality of the engineered features or patterns the system learns from the data determines the accuracy of prediction or classification. Moreover, it involves tedious parameter tunings, which tends to an entirely heuristic approach. In deep learning, the features can be learnt optimally from the data automatically, and these features can be interpretable by visualization [46]. Although, several hyperparameters need to be set, including the number of layers, number neurons, activation functions, etc. A tedious grid search approach makes has to be implemented to determine optimal hyperparameters.  

# D. Applied to App on Mobiles  

The proposed CRNN is an adaptive algorithm for different diabetic subjects. The property of adapte lies in two aspects. Firstly, it is data driven and individualised, thus different subjects have different models. Secondly, the model can be continuously evolving while people are using it. In details, the model is saved as a trained neural network. We use Keras sequential model with Tensorflow backend to train the neural network, and the result model can be easily saved as a single file. This file can be compiled as a “.tflite” or a “.pb” file to the app on mobiles, by using a Tensorflow Lite converter. The model file can be upload, upload and updated continuously at the cloud. Then the app may use it to provide the predictive glycaemic curve and show it the screen. A demonstration on the Android system is shown in Figure 7 In addition, we also found that the execution time of the model is 6ms on a Android phone (LG Nexus5 with Processor:2.26GHz quad-core, RAM:2GB) and 780ms on a laptop (MacPro with Processor: 3.1GHz Intel Core i5, RAM:8GB), though laptops usually have better processing units. The reasons might be the quantisation of weights and biases (e.g. 8 bit integer vs. 32 bit floating point) leading to simpler and faster computation at each layer.  

![](images/9556eb3bd5f19bede0450878591c34fbc0a0f88bcc2e79238f73fed4a194518f.jpg)  
Fig. 7. An illustration of the glucose level shown in an app interface on an Android system, where the red curve is the historic blood glucose, black dash line is the current time, and the red dot curve is the prediction provided by the model.  

# E. Limitations  

Though the CRNN has achieved very good accuracy in prediction, some challenges exist that can form the basis of further research. The underlying assumption of the deep learning approach, and indeed in the supervised learning paradigm, is that data is independent and identically distributed. Given that learning is solely based on historical data, unexpected predictions may occur given that correlations learned in the data may not imply causation. Thus hybrid approaches whereby the deep learning model is used to make an accurate prediction, and rules of meal/bolus supported by physiological model avoid apparent errors that might result. Based on the CRNN approach proposed in this paper, it is possible to develop the hybrid method, which may have the advantages of both conventional and DL algorithms.  

# V. CONCLUSION  

In this paper a convolutional recurrent neural network was proposed as an effective method for blood glucose prediction. The architecture of the neural network includes a multilayer CNN followed by a modified RNN, where the CNN can capture the features or patterns of the multi-dimensional time series data properly, and the modified RNN is capable of analyzing the previous sequential data and provide the predictive glucose level further. The method can train models for each diabetic subject using their own data, thus the model for the prediction is adaptive and specialized for each person. After obtaining the trained neural network, it can be applied locally or on portable devices, which can then be implemented widely in the daily life. It has been found that the proposed method reduces the RMSE, MARD in the in silico experiments. While in the test using the clinical data, the proposed method performs quite well in most cases comparing to other latest approaches, and shows competitive performance in forecasting hypoglycaemia and hyperglycaemia.  

# VI. APPENDIX  

The authors thank EPSRC, ARISES project for support and providing the clinical data. We would also like to thank Prof.  

R. Spence, T. Zhu and C. Demasson’s for the contribution to the mobile app interface design.  

TABLE VCOMPARISON OF DIFFERENT PREDICTION METHODS FOR THE10-ADULT VIRTUAL DIABETIC SUBJECTS CONSIDERED INTERMS OF RMSE(mg/dL)  

![](images/b7ec12e2cd856522d8a44b4e0012c11e7fecabf6f18d6624c43a248971876028.jpg)  

TABLE VICOMPARISON OF DIFFERENT PREDICTION METHODS FOR THE10-ADULT VIRTUAL DIABETIC SUBJECTS CONSIDERED INTERMS OF MARD $\%)$  

![](images/724688f4c615d7711ec3fdff111e70beb2145ec18431fce8d0a3f859cfce3100.jpg)  

TABLE VII DIFFERENT PREDICTION METHODS COMPARISON FOR THE CLINICAL 5 DIABETIC SUBJECTS CONSIDERED IN TERMS OF RMSE (mg/dL)   

![](images/aefd59cbf9e835672bd2648c034ac4ea6788c1e1e78e717f7a5c80c6b7e99998.jpg)  

# REFERENCES  

[1] S. Oviedo, J. Veh, R. Calm, and J. Armengol, “A review of personalized blood glucose prediction strategies for t1dm patients,” International Journal for Numerical Methods in Biomedical Engineering, vol. 33, no. 6, p. 2833, 2017. [Online]. Available: https://onlinelibrary.wiley. com/doi/abs/10.1002/cnm.2833  

[2] M. Vettoretti, A. Facchinetti, G. Sparacino, and C. Cobelli, “Type 1 diabetes patient decision simulator for in silico testing safety and effectiveness of insulin treatments,” IEEE Transactions on Biomedical Engineering, pp. 1–1, 2018.  

TABLE VIII   
DIFFERENT PREDICTION METHODS COMPARISON FOR THE   
CLINICAL 5 DIABETIC SUBJECTS CONSIDERED IN TERMS OF MARD (%)   
[3] P. Pesl, P. Herrero, M. Reddy, M. Xenou, N. Oliver, D. Johnston, C. Toumazou, and P. Georgiou, “An advanced bolus calculator for type 1 diabetes: System architecture and usability results,” IEEE Journal of Biomedical and Health Informatics, vol. 20, no. 1, pp. 11–17, Jan 2016. [4] A. Facchinetti, S. Favero, G. Sparacino, and C. Cobelli, “An online failure detection method of the glucose sensor-insulin pump system: Improved overnight safety of type-1 diabetic subjects,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 2, pp. 406–416, Feb. 2013. [5] S. Zavitsanou, A. Mantalaris, M. C. Georgiadis, and E. N. Pistikopoulos, “In silico closed-loop control validation studies for optimal insulin delivery in type 1 diabetes,” IEEE Transactions on Biomedical Engineering, vol. 62, no. 10, pp. 2369–2378, Oct. 2015. [6] K. van Heusden, E. Dassau, H. C. Zisser, D. E. Seborg, and F. J. D. III, “Control-relevant models for glucose control using a priori patient characteristics,” IEEE Transactions on Biomedical Engineering, vol. 59, no. 7, pp. 1839–1849, Jul. 2012. [7] M. M. Ahmadi and G. A. Jullien, “A wireless-implantable microsystem for continuous blood glucose monitoring,” IEEE Transactions on Biomedical Circuits and Systems, vol. 3, no. 3, pp. 169–180, Jun. 2009. [8] A. Facchinetti, “Continuous glucose monitoring sensors: Past, present and future algorithmic challenges,” Sensors, vol. 16, no. 12, 2016.   
[9] A. Revert, F. Garelli, J. Pic, H. D. Battista, P. Rossetti, J. Vehi, and J. Bondia, “Safety auxiliary feedback element for the artificial pancreas in type 1 diabetes,” IEEE Transactions on Biomedical Engineering, vol. 60, no. 8, pp. 2113–2122, Aug. 2013.   
[10] Gutierrez-Osuna and N. HT., “A method for evaluating datapreprocessing techniques for odour classification with an array of gas sensors,” IEEE Trans Syst Man Cybern B Cybern., vol. 29, no. 5, pp. 626 – 632, 1999.   
[11] E. I. Georga, V. C. Protopappas, D. Ardig, M. Marina, I. Zavaroni, D. Polyzos, and D. I. Fotiadis, “Multivariate prediction of subcutaneous glucose concentration in type 1 diabetes patients based on support vector regression,” IEEE Journal of Biomedical and Health Informatics, vol. 17, no. 1, pp. 71–81, Jan 2013.   
[12] K. Yan and D. Zhang, “Blood glucose prediction by breath analysis system with feature selection and model fusion,” in 36th Annual International Conference of the IEEE Engineering in Medicine and Biology Society, Aug. 2014, pp. 6406–6409.   
[13] S. Bedoui, R. Faleh, H. Samet, and A. Kachouri, “Electronic nose system and principal component analysis technique for gases identification,” in 10th International Multi-Conferences on Systems, Signals Devices 2013 (SSD13), Mar. 2013, pp. 1–6.   
[14] A. Loutf,i S. Coradeschi, G. K. Mani, P. Shankar, and J. B. B. Rayappan, “Electronic noses for food quality: A review,” Journal of Food Engineering, vol. 144, pp. 103 – 111, 2015. [Online]. Available: http://www.sciencedirect.com/science/article/pii/S0260877414003276   
[15] C. Pe´rez-Gandı´a, A. Facchinetti, G. Sparacino, C. Cobelli, E. G´omez, M. Rigla, A. de Leiva, and M. Hernando, “Artificial neural network algorithm for online glucose prediction from continuous glucose monitoring,” Diabetes technology & therapeutics, vol. 12, no. 1, pp. 81–88, 2010.   
[16] C. Zecchin, A. Facchinetti, G. Sparacino, G. D. Nicolao, and C. Cobelli, “Neural network incorporating meal information improves accuracy of short-time prediction of glucose concentration,” IEEE Transactions on Biomedical Engineering, vol. 59, no. 6, pp. 1550–1560, Jun. 2012.   
[17] K. Plis, R. Bunescu, C. Marling, J. Shubrook, and F. Schwartz, “A machine learning approach to predicting blood glucose levels for diabetes management,” in Modern Artificial Intelligence for Health Analytics Papers from the AAAI-14.   
[18] M. v. d. W. H.N. Mhaskar, S.V. Pereverzyev, “A deep learning approach to diabetic blood glucose prediction,” https://arxiv.org/abs/1707.05828, 2017.   
[19] C. Marling and R. Bunescu, “The OhioT1DM dataset for blood glucose level prediction,” in The 3rd International Workshop on Knowledge Discovery in Healthcare Data, Stockholm, Sweden, July 2018, CEUR proceedings in press, available at http://smarthealth.cs.ohio.edu/bglp/OhioT1DM-dataset-paper.pdf.   
[20] Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell, “Caffe: Convolutional architecture for fast feature embedding,” in Proceedings of the 22Nd ACM International Conference on Multimedia, ser. MM ’14. New York, NY, USA: ACM, 2014, pp. 675–678. [Online]. Available: http://doi.acm.org/10.1145/2647868.2654889   
[21] D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, I. A. Julian Schrittwieser, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis, “Mastering the game of go with deep neural networks and tree search,” Nature, vol. 529, p. 484489, Jan. 2016.   
[22] G. Litjens, C. I. Snchez, N. Timofeeva, M. Hermsen, I. Nagtegaal, I. Kovacs, C. H. van de Kaa, P. Bult, B. van Ginneken, and J. van der Laak, “Deep learning as a tool for increased accuracy and efficiency of histopathological diagnosis,” Scientific Reports, vol. 6, p. 26286, May 2016.   
[23] J. Schmidhuber, “Deep learning in neural networks: An overview,” Neural Networks, vol. 61, pp. 85 – 117, 2015. [Online]. Available: http://www.sciencedirect.com/science/article/pii/S0893608014002135   
[24] R. Miotto, F. Wang, S. Wang, X. Jiang, and J. T. Dudley, “Deep learning for healthcare: review, opportunities and challenges,” Briefings in Bioinformatics, p. 111, 2017. [Online]. Available: http://dx.doi.org/10.1093/bib/bbx044   
[25] K. Li and P. H. Y. G. P. G. F. Liu, H. Dong, “A deep learning platform for diabetes big data analysis,” in 11th Advanced Technologies and Treatments for Diabetes. Mary Ann Liabery, 2018, pp. A116–A116.   
[26] T. Zhu, K. Li, P. Herrero, J. Chen, and P. Georgiou, “A deep learning algorithm for personalized blood glucose prediction.”   
[27] Y. Bengio, “Deep learning of representations: Looking forward,” in Statistical Language and Speech Processing. Berlin, Heidelberg: Springer Berlin Heidelberg, 2013, pp. 1–37.   
[28] Q. Zhang, Y. N. Wu, and S.-C. Zhu, “Interpretable convolutional neural networks,” in The IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2018, pp. 8827–8836.   
[29] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. The MIT Press, 2016.   
[30] M. Abadi, A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man´e, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Vie´gas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng, “TensorFlow: Large-scale machine learning on heterogeneous systems,” 2015, software available from tensorflow.org. [Online]. Available: https://www.tensorflow.org/   
[31] C. Zhao, E. Dassau, L. Jovanovic, H. C. Zisser, I. Francis J. Doyle, and D. E. Seborg, “Predicting subcutaneous glucose concentration using a latent-variable-based statistical method for type 1 diabetes mellitus,” Journal of Diabetes Science and Technology, vol. 6, no. 3, pp. 617–633, 2012, pMID: 22768893.   
[32] D. A. Finan, F. J. Doyle III, C. C. Palerm, W. C. Bevier, H. C. Zisser, L. Jovanoviˇc, and D. E. Seborg, “Experimental evaluation of a recursive model identification technique for type 1 diabetes,” Journal of diabetes science and technology, vol. 3, no. 5, pp. 1192–1202, 2009.   
[33] G. Sparacino, F. Zanderigo, S. Corazza, A. Maran, A. Facchinetti, and C. Cobelli, “Glucose concentration can be predicted ahead in time from continuous glucose monitoring sensor time-series,” IEEE Transactions on Biomedical Engineering, vol. 54, no. 5, pp. 931–937, May 2007.   
[34] S. G. Mougiakakou, A. Prountzou, D. Iliopoulou, K. S. Nikita, A. Vazeou, and C. S. Bartsocas, “Neural network based glucose insulin metabolism models for children with type 1 diabetes,” in 2006 International Conference of the IEEE Engineering in Medicine and Biology Society, Aug. 2006, pp. 3545–3548.   
[35] M. Reddy, P. Pesl, M. Xenou, C. Toumazou, D. Johnston, P. Georgiou, P. Herrero, and N. Oliver, “Clinical safety and feasibility of the advanced bolus calculator for type 1 diabetes based on case-based reasoning: A 6-week nonrandomized single-arm pilot study,” Diabetes Technology & Therapeutics, vol. 18, no. 8, pp. 487–493, 2016, pMID: 27196358. [Online]. Available: https://doi.org/10.1089/dia.2015.0413   
[36] R. Atanassov, P. Bose, M. Couture, A. Maheshwari, P. Morin, M. Paquette, M. Smid, and S. Wuhrer, “Algorithms for optimal outlier removal,” Journal of Discrete Algorithms, vol. 7, no. 2, pp. 239 – 248, 2009, selected papers from the 2nd Algorithms and Complexity in Durham Workshop ACiD 2006. [Online]. Available: http://www.sciencedirect.com/science/article/pii/S1570866709000021   
[37] H. Shum, K. Ikeuchi, and R. Reddy, Principal Component Analysis with Missing Data and Its Application to Polyhedral Object Modeling. Boston, MA: Springer US, 2001, pp. 3–39. [Online]. Available: https://doi.org/10.1007/978-1-4615-0797-0 1   
[38] G. E. Batista and M. C. Monard, “An analysis of four missing data treatment methods for supervised learning,” Applied Artificial Intelligence, vol. 17, no. 5-6, pp. 519–533, 2003. [Online]. Available: https://doi.org/10.1080/713827181   
[39] D. Li, J. Deogun, W. Spaulding, and B. Shuart, “Towards missing data imputation: A study of fuzzy k-means clustering method,” in Rough Sets and Current Trends in Computing, S. Tsumoto, R. Słowi´nski, J. Komorowski, and J. W. Grzymała-Busse, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004, pp. 573–579.   
[40] L. Lekha and S. M, “Real-time non-invasive detection and classification of diabetes using modified convolution neural network,” IEEE Journal of Biomedical and Health Informatics, pp. 1–1, 2018.   
[41] S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, 1997.   
[42] A. Graves, A. R. Mohamed, and G. Hinton, “Speech recognition with deep recurrent neural networks,” in IEEE Int. Conf. on Acou., Spe. and Sig. Proc., 2013, p. 6645.   
[43] W. Zaremba, I. Sutskever, and O. Vinyals, “Recurrent neural network regularization,” 2014, cite arxiv:1409.2329. [Online]. Available: http://arxiv.org/abs/1409.2329   
[44] N. Srivastava, G. E. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, “Dropout: a simple way to prevent neural networks from overfitting.” Journal of Machine Learning Research, vol. 15, no. 1, pp. 1929–1958, 2014.   
[45] “The uva/padova type 1 diabetes simulator,” Jounral of Diabetes Sci Technol., vol. 8, no. 1.   
[46] G. Montavon, W. Samek, and K.-R. Mller, “Methods for interpreting and understanding deep neural networks,” Digital Signal Processing, vol. 73, pp. 1 – 15, 2018. [Online]. Available: http://www.sciencedirect. com/science/article/pii/S1051200417302385  

![](images/097a9a2770042ee59df734cf0763c6f2e374b856bea99295ffcaddd293e923e3.jpg)  