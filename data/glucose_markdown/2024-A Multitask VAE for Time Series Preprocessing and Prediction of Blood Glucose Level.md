# A Multitask VAE for Time Series Preprocessing and Prediction of Blood Glucose Level  

Ali ABUSALEH Air Liquide R&D Les Loges-en-Josas 78350, France abusalehali@icloud.com  

Mehdi RAHIM Air Liquide R&D Les Loges-en-Josas 78350, France mehdi.rahim@airliquide.com  

Abstract—Data preprocessing is a critical part of time series data analysis. Data from connected medical devices often have missing or abnormal values during acquisition. Handling such situations requires additional assumptions and domain knowledge. This can be time-consuming, and can introduce a significant bias affecting predictive model accuracy and thus, medical interpretation. To overcome this issue, we propose a new deep learning model to mitigate the preprocessing assumptions. The model architecture relies on a variational auto-encoder (VAE) to produce a preprocessing latent space, and a recurrent VAE to preserve the temporal dynamics of the data. We demonstrate the effectiveness of such an architecture on telemonitoring data to forecast glucose-level of diabetic patients. Our results show an improvement in terms of accuracy with respect of existing state-of-the-art methods and architectures.  

Index Terms—Generative Deep Learning, Data Preprocessing, Time Series, VAE.  

of the underlying physiological mechanisms. However, a comprehensive understanding of the data and proper preprocessing are crucial. In particular, data preprocessing has a significant impact on the quality of the data-driven forecasting model of glucose-level. Yet, there are no standard preprocessing pipelines for CGM readings.  

To mitigate the preprocessing assumptions, we propose a new deep learning model. The model architecture relies on a variational auto-encoder (VAE) to produce a preprocessing latent space, and a recurrent VAE to preserve the temporal dynamics of the data. We apply this model in the context of blood glucose forecast for patients with Diabetes. Our results highlight the effectiveness of our approach compared to baselines and state-of-the art predictive models. Our technical contributions are two-fold :  

# I. INTRODUCTION  

Diabetes mellitus (DM) is a complex and pervasive metabolic disorder characterized by strong variations of the blood glucose (BG) level. The physiopathology is complex and mainly caused by defects in insulin secretion, insulin action, or both. There are several types of diabetes, including type 1 diabetes mellitus (T1DM) where the patient has a complete lack of insulin production, type 2 diabetes mellitus (T2DM) where the body’s cells become less responsive to insulin, and relative insulin deficiency, where the pancreas fails to produce enough insulin to compensate for insulin resistance, and other less common forms.  

Glucose-level forecasting from connected continuous glucose monitoring (CGM) devices help anticipate hyperglycmia (high glucose level) or hypoglycemia (low glucose level). CGM can measure glucose level every five minutes, enabling an almost real time remote monitoring of the patient health status. The taxonomy of blood glucose level forecasting can be categorized into two main approaches [5]: i) Physiological models, which requires an in-depth understanding of an individual’s physiological mechanisms. This approach involves the study of how the human metabolism works and how it affects blood glucose levels. This often referred to as white-box [9]; ii) Data-driven models, which utilizes historical data as a foundation for predictions. This approach is often referred to as a black-box [9] model. It does not require an understanding • Architecture: A novel VAE architecture that incorporates a temporal attention mechanism. This architecture is a self-contained preprocessing model and a prediction network. • Loss: Optimized loss functions tailored for enhancing forecasting accuracy in healthcare applications.  

The paper is organized as follows : section II introduces the methods related to time series forecasting with deep learning models. Section III describe our contribution and architecture. Section IV shows the experiments and results. Section $\mathrm{v}$ concludes the paper and discusses future works.  

# II. DEEP LEARNING FOR TIME SERIES  

Time series are a widespread data modality to characterize phenomena in numerous domains such as finance, healthcare, and industry. Proper handling and preprocessing of these data are crucial to ensure accurate and reliable analyses and predictions. Preprocessing techniques for time series data encompass different items such as missing values, outliers, and feature scaling. If uncorrected, time series analysis can be significantly distorted. Missing values in time series data are often due to sensor malfunctions or data transmission errors. Techniques such as Multiple Imputation (MI), are effective in addressing these issues [24]. Similarly, outlier detection methods, help identify and mitigate the impact of anomalous data points that could skew results [24], normalization, Feature extraction and selection, and many other factors play a pivotal role in the model outcome. Time series preprocessing includes techniques such as normalization ( [10], [11]), scaling [10], smoothing [12], approximation, interpolations. Each step has an impact on the final representation of the time series. Generative models, such as Variational Autoencoders (VAEs), have shown promise in representing data in a latent space [13]. While traditional VAEs have demonstrated their effectiveness in representation learning, they suffer from limitations in handling time series data. Recent advances have proposed novel VAE architectures to address these limitations. For example, [15] proposed a VAE with a recurrent neural network (RNN) encoder and decoder, while [16] proposed a VAE with a temporal convolutional network (TCN) encoder and decoder. These architectures have shown improved performance in representing time-series data compared to traditional VAEs. In the context of blood glucose level forecasting, VAEs have shown potential for handling missing data and generating synthetic data. For instance, [17] use a VAE to impute missing clinical data in electronic health records, while [18] use a VAE to generate synthetic blood glucose data for a population of patients. However, blood glucose level forecasting presents unique challenges that traditional VAEs may not fully address. Factors such as nutrition and physical exercises, and the irregular usage of CGM devices, complicate the forecasting task. To overcome this, we propose a novel VAE architecture that incorporates a temporal attention mechanism. This enables the VAE to better capture the dynamics, performing a preprocessing and improve forecasting accuracy. We introduce in the following section our contribution. To the best of our knowledge, this model is an original architecture tailored for both time series preprocessing and forecasting.  

# III. MODEL ARCHITECTURE  

We describe in this section our model to reduce the dependency on preprocessing techniques while maintaining the main objective of long-term forecasting and accuracy. This model combines different architectures as discussed in [17] and [15]. Our proposed model leverages the Variational Autoencoder (VAE) framework to encode time series data into a latent space. Additionally, we integrate Recurrent Neural Networks (RNNs) within the VAE architecture to preserve the temporal dependencies inherent in the sequential data. This hybrid approach not only ensures robust data imputation but also enhances the accuracy of long-term forecasting. It captures both individual data point nuances and overall temporal patterns. We detail below the main components of the model  

# A. Data in the latent space  

Mathematical Formulation: The VAE represents the data generation process using a probabilistic framework. The encoder maps the input $\mathbf{x}$ to a probability distribution over the latent space, $q_{\phi}(\mathbf{z}|\mathbf{x})$ , where $\phi$ are the parameters of the encoder. This distribution is typically chosen to be a Gaussian distribution with mean $\mu$ and variance $\sigma^{2}$ .  

$$
q_{\phi}(\mathbf{z}|\mathbf{x})=\mathcal{N}(\mathbf{z};\mu(\mathbf{x}),\sigma^{2}(\mathbf{x}))
$$  

The decoder then maps the latent variable ${\bf z}$ back to a distribution over the input space, $p_{\theta}(\mathbf{x}|\mathbf{z})$ , where $\theta$ are the parameters of the decoder. The reconstruction of the input is  

$$
p_{\theta}(\mathbf{x}|\mathbf{z})=\mathcal{N}(\mathbf{x};\hat{\mathbf{x}},\sigma_{x}^{2})
$$  

To train the VAE, we maximize the evidence lower bound (ELBO) on the marginal likelihood of the data. The ELBO consists of two terms: a reconstruction loss and a regularization term. The reconstruction loss measures how well the decoder can reconstruct the input data, while the regularization term enforces a prior distribution $p(\mathbf{z})$ (typically a standard normal distribution) on the latent variables.  

$$
\mathcal{L}(\phi,\theta;\mathbf{x})=\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}\mathrm{[log}\,p_{\theta}(\mathbf{x}|\mathbf{z})]-D_{\mathrm{KL}}(q_{\phi}(\mathbf{z}|\mathbf{x})||p(\mathbf{z}))
$$  

Here, $\mathbb{E}_{q_{\phi}(\mathbf{z}|\mathbf{x})}[\log p_{\theta}(\mathbf{x}|\mathbf{z})]$ is the expected reconstruction loss and $D_{\mathrm{KL}}$ is the Kullback-Leibler divergence, which measures the difference between the learned latent distribution $q_{\phi}(\mathbf{z}|\mathbf{x})$ and the prior distribution $p(\mathbf{z})$ .  

# B. Temporal Dynamics in Latent Space  

While a standard VAE offers advantages in the latent space representation, it may not effectively capture the temporal dependencies inherent in sequential data like time series. To address this limitation, we incorporate an RNN as an encoderdecoder component of the VAE [15]. This modification aims to preserve the temporal dynamics of the data, allowing the model to better capture the sequential dependencies present in time series data.  

The Variational Recurrent Neural Network (VRNN) [15] integrates a VAE at each time step $t$ conditioned on the previous RNN state $h_{t-1}$ . This configuration enables the VAE to account for the sequential nature of the data more effectively than a traditional VAE. Unlike a standard VAE, the latent variable prior in VRNN follows a non-standard Gaussian distribution, and the generating distribution is conditioned not only on $z_{t}$ but also on $h_{t-1}$ . The RNN updates its hidden state $h_{t}$ using a recurrence equation that incorporates $\phi_{x}^{\tau},\ \phi_{z}^{\tau}$ , and $h_{t-1}$ . This setup defines the distributions $p(z_{t}|x_{<t},z_{<t})$ and $p(x_{t}|\boldsymbol{z}_{\le t},\boldsymbol{x}_{<t})$ . Similarly, the approximate posterior $q(z_{t}|x_{\le t},z_{<t})$ is conditioned on $x_{t}$ and $h_{t-1}$ . The encoding of the posterior and decoding for generation are linked through $h_{t-1}$ , resulting in the factorization [15]  

$$
q(\boldsymbol{z}_{\le T}|\boldsymbol{x}_{\le T})=\prod_{t=1}^{T}q(\boldsymbol{z}_{t}|\boldsymbol{x}_{\le t},\boldsymbol{z}_{<t})
$$  

# C. A synthetic example  

The latent space in a VAE [17] can be used to better capture the data dynamics while preserving the individuality of each data point. The following example shows three different time series explained as follows:  

# • $t s_{1}=\sin(2\pi\cdot t i m e/12)+0.5\cdot\mathcal{N}(n_{\mathrm{samples}},1)$ • $t s_{2}=\sin(2\pi\cdot t i m e/6)+0.5\cdot{\mathcal N}(n_{\mathrm{samples}},1)$ • $t s_{3}=\sin(2\pi\cdot t i m e/4)+0.5\cdot{\mathcal N}(n_{\mathrm{samples}},1)$  

where $t s_{1}$ and $t s_{2}$ retain all the data points, $t s_{3}$ is masked for an interval of 1 hour (60 data points) to simulate a missing data scenario. Figure 1 shows the visual representation of the three time series. After projecting the time series into a latent space using the VAE encoder, even with missing data points, the model can learn the relationships between existing data and use them to infer the missing information. Here’s the equation for projecting a time series data point $(x)$ into the latent space using the VAE encoder: $z={\mathrm{encoder}}(x)$ . This equation represents encoding the data point $x$ using the encoder function of the VAE to obtain its latent representation $z$ . By analyzing the relationships between latent representations in the latent space, the model can potentially estimate missing values for incomplete time series.  

![](images/b75f26b0f7cc2a277878ddd1eee157ba0e5a156eb68efc7abcb683f5340e59c9.jpg)  
Fig. 1: Synthetic time series to highlight the VAE based imputation.  

The masked time series $(t s_{3})$ is affected by the dynamics of $t s_{1}$ and $t s_{2}$ , and a simulation of contextual data imputation is performed on $t s_{3}$ . The following figure 2 shows the imputation in $t s_{3}$ .  

![](images/bbc56dfbf52fefb66587f33d2531f1e95d2fb4fd2bc6bc2d29e4b690be05b89d.jpg)  
Fig. 2: Comparison of $t s_{3}$ before and after projection to latent space  

This approach offers a promising solution for handling missing data in time series forecasting by reducing reliance on manual intervention and potentially improving the accuracy of long-term forecasts. However, further research is needed to explore the effectiveness of different imputation techniques within the latent space and their  

# D. Proposed hybrid ML Model/Architecture  

We propose a tailored version of VAE, where we have incomplete data scarcity. We selected this architecture component after benchmarking different RNN components. This benchmark is discussed in IV.  

Architecture: The architecture of the VAE involves an RNN-based encoder and decoder. The encoder maps the input x to a latent space, represented by the mean $(\mu)$ and the logarithm of the variance $(\log\sigma^{2})$ . The latent variable $\mathbf{z}$ is then sampled and passed through the decoder, which reconstructs the input features and predicts future glucose levels. This architecture also functions as a data imputation model, where the RNN components capture the temporal dynamics, and the latent space captures the correlations in the data, effectively handling missing periods. The components are as follows:  

• An RNN encoder with input size $d$ and hidden size $h$ . • Fully connected layers to compute the mean $(\mu)$ and log variance $(\log\sigma^{2})$ . • An RNN decoder to reconstruct the input and predict future values. • Separate output layers for reconstruction and prediction. The figure 3 shows an LSTM version of the proposed rchitecture.  

Objective: The model aims to learn a latent space representation that encapsulates the seasonal patterns, like patient’s metabolism. The model also performs data preprocessing such as imputation, etc, and forecasting within a time horizon. This objective is achieved by maximizing the likelihood of the target next inference window $w$ , given the input inference window $\mathbf{x}$ . Table I summarizes the notations used to describe the optimization.  

TABLE I: Notations used and their meanings   

![](images/2cbec334f1182c423d0febf34f9ca5450ab7cf4ccff6ba720375053216942de8.jpg)  

The optimization process aims to maximize the evidence lower bound (ELBO) objective:  

$$
\begin{array}{r l}&{\mathrm{ELBO}(\pmb{\theta})=\mathbb{E}_{q_{\pmb{\theta}}(\mathbf{z}|\mathbf{x},h_{t-1})}\left[\log p_{\pmb{\theta}}(\mathbf{x}\mid\mathbf{z},h_{t-1})\right]}\\ &{\quad\quad\quad\quad\quad-\mathrm{KL}\left(q_{\pmb{\theta}}(\mathbf{z}\mid\mathbf{x},h_{t-1})\|p(\mathbf{z})\right)}\end{array}
$$  

![](images/2b6247d27355e4716495401518f3a2db61bdd11b923db5934cc52ef7346f74b5.jpg)  
Fig. 3: Patient-specific VAE-LSTM architecture. The architecture consists of a fully connected layer, followed by two main components: an encoder and a decoder, both built based on GRU/LSTM. The encoder maps the input data $\mathbf{x}$ to a latent representation $\mathbf{z}$ , while the decoder reconstructs the best representative of the data from this latent representation. The missing data is represented and interpolated by the latent space.  

where $\mathrm{KL}(q_{\theta}(\mathbf{z}\mid\mathbf{\theta}\propto,h_{t-1})\|p(\mathbf{z}))$ denotes the KullbackLeibler (KL) divergence between the encoder’s posterior distribution and the prior distribution $p(\mathbf{z})$ .  

Loss Function: The loss function for the model comprises three components:  

1) The reconstruction loss ${\mathcal{L}}_{\mathrm{reco}}$ quantifies the discrepancy between the reconstructed features and the actual input features, using Mean Squared Error (MSE).  

$$
\mathcal{L}_{\mathrm{reco}}=\frac{1}{n}\sum_{i=1}^{n}\|\mathbf{x}_{i}-\mathbb{E}_{q_{\theta}(\mathbf{z}|\mathbf{x}_{i},h_{t-1})}[p_{\theta}(\mathbf{x}_{i}\mid\mathbf{z},h_{t-1})]\|^{2}
$$  

2) The prediction loss $\mathcal{L}_{\mathrm{pred}}$ measures the accuracy of predicted future glucose levels against actual values, using Mean Squared Error (MSE).  

$$
\mathcal{L}_{\mathrm{pred}}=\frac{1}{n}\sum_{i=1}^{n}\|\mathbf{y}_{i}-\mathbb{E}_{q_{\theta}(\mathbf{z}|\mathbf{x}_{i},h_{t-1})}[p_{\theta}(\mathbf{y}_{i}\mid\mathbf{z},h_{t-1})]\|^{2}
$$  

3) The $K L$ divergence loss $\mathcal{L}_{\mathrm{KL}}$ measures the divergence between the learned latent distribution and the prior distribution.  

$$
\mathcal{L}_{\mathrm{KL}}=\frac{1}{n}\sum_{i=1}^{n}-0.5\sum_{j}\left(1+\log(\sigma_{j}^{2})-\mu_{j}^{2}-\sigma_{j}^{2}\right)
$$  

The total loss is  

$$
\mathcal{L}_{\mathrm{total}}=\alpha\mathcal{L}_{\mathrm{reco}}+\beta\mathcal{L}_{\mathrm{pred}}+\gamma\mathcal{L}_{\mathrm{KL}}.
$$  

One should note that any kind of RNN can be used, such as LSTM, GRU, etc.  

# IV. EXPERIMENTS AND RESULTS  

We present the experiments on forecasting blood glucose levels over a limited time-series dataset. We analyze : i) The performance of the proposed model compared to state-of-theart models and baselines, the analysis included a mathematical metrics (RMSE, MAPE, nMAPE), and clinical aspect of the model (Clarke Error Grid). We evaluate the predictive models on the public dataset OhioT1DM 2018 [22]. The dataset contains 8 weeks of CGM, insulin, and self-reported meals for a total of 6 patients with type-1 diabetes. The dataset is primarily utilized for blood glucose prediction.  

A. Glucose-level prediction accuracy  

We compare our model to different RNN models (LSTM, Bi-LSTM, GRU, Bi-GRU), in addition to a statistical model ARIMA and two baselines from naive approaches: forward fill, linear trend. These models are trained on all patient data with an $80\%-20\%$ training-testing split and 20 epochs.  

Table II depicts the mean $\pm$ standard deviation of RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error), and nMAPE (Normalized Mean Absolute Percentage Error) for a prediction horizon of 30 minutes (6 steps). Table III provides the results for a prediction horizon of 1 hour (12 steps). Results show that Bi-GRU models while demonstrating good overall performance in terms of RMSE and MAPE, are suboptimal for our specific glucoselevel prediction use case. Conversely, our proposed VAEGRU model (highlighted in yellow in Table III) shows higher performance. It achieves the lowest RMSE and MAPE values among all models evaluated for the 1-hour prediction horizon.  

TABLE II: Forecasting accuracy comparison (30 minutes)   

![](images/36b7e46e6dab630b076d1d26a2ff2ab5f34db4b4838bde083edbbb539d18a118.jpg)  

TABLE III: Forecasting accuracy comparison (1 Hour)   

![](images/4e86f3e0c547dc5f7990f4d1c3a0d7c2263464d87037c96f9b2721cd9943bba7.jpg)  

Clinical performance of glucose-level prediction: In addition to the statistical metrics, we analyze the performance of the predictive models in terms of diabetes-related metrics called Clark Error Grid. The Clarke Error Grid is a visual tool used to assess the accuracy of blood glucose monitoring systems or continuous glucose monitoring systems. It is also used to assess glucose-level forecasting. The grid is divided into five zones: A to E. Each zone represents a different level of accuracy and potential clinical risk. Zone A is the most accurate, while zone E is an extremely large overestimation or underestimation of the true glucose value. They could lead to treatment decisions that have serious medical risk.  

As shown in Table IV, VAE-RNN models exhibit better clinical outcomes. Specifically, VAE-LSTM, while quantitatively superior in most metrics, shows less reliability in clinical Zone-D. Bi-RNN models have a negative impact on clinical results. Table IV presents the clinical metrics including A, B, C, D, and E, where higher values of $\mathrm{~A~}+\mathrm{~B~}$ indicate better performance (aiming for $>90\%$ ), and lower values of $\boldsymbol{\mathrm{D}}+\boldsymbol{\mathrm{E}}$ are desired. Based on the clinical metrics presented in Table IV, the VAE-GRU model (highlighted in yellow) achieves the highest values for A $83.55~\pm~7.64)$ and B ( $14.19\,\pm$ 6.45), indicating superior performance in predicting glucose levels above $90\%$ . It also shows low values for $\mathrm{D}$ ( $1.74~\pm$ 1.26) and E $[0.09\,\pm\,0.14)$ , demonstrating minimal errors in critical glucose level predictions, which are crucial for clinical management. In contrast, while VAE-LSTM performs well in  

A and B metrics, it shows higher values in D $(7.54\,\pm\,8.75)$ and E $(0.04\,\pm\,0.06)$ , suggesting potential clinical limitations compared to VAE-GRU.  

TABLE IV: Clarke Error Grid model comparison (1 Hour)   

![](images/272e959f94be0f43f3216b89796af55a95b14622eacf19ecb1e75c47dc61512b.jpg)  

We show in figure 4 one example of Clarke Error Grid. The first row shows GRU models, both GRU and biGRU, while the second row displays the (Bi)-LSTM models. The results indicate that LSTM models exhibit resistance to predictions above 160, whereas biLSTM models have a higher threshold of around 200 with adverse clinical effects. Similar observations apply to GRU models. Our proposed models demonstrate a balanced approach, achieving higher prediction thresholds while maintaining superior clinical performance.  

![](images/025671b18e8826713135883dd9c64f11879f7a35f2cc0d41409fee35e3d2dba2.jpg)  
Fig. 4: Clarke Error Grid on Patient-563. LSTM models exhibit resistance to predictions above 160, while biLSTM models have a higher threshold around 200 with adverse clinical effects. Similar observations apply to GRU models. Our proposed models demonstrate a balanced approach, achieving higher prediction thresholds while maintaining superior clinical performance.  

Long-term prediction horizon: We tested the proposed model in the long-term, namely 240 data points (4 hours)in the future). Table V shows the Normalized Mean Absolute Percentage Error (nMAPE) values for different models at various prediction intervals. The results for VAE-LSTM and VAE-GRU (ours) are also included for comparison with other models, and they suggest better results in the long term with better stability over a longer horizon.  

# B. Learning speed of predictive models  

Beyond the accuracy of the predictive models, we assess the speed of convergence of our models compared to classical RNNs. We compare the proposed model in terms of resources and training time, as shown in figure 5. The proposed model demonstrates faster learning, averaging 7 epochs to converge, while other models tend to require longer training times to achieve similar results. This efficiency makes our proposed model not only effective in clinical metrics but also resourceefficient.  

TABLE V: Long-term prediction horizon comparison   

![](images/2c14a0bc8db9f23345bd8fc71784d08d766f919faa5192bcd8fa0681f5ed4e0d.jpg)  

![](images/3bc5eb30cc3f6c9dfe9990aaf7e462cc8338914c878e0b1a1baff4161465e4ec.jpg)  
Fig. 5: Performance of different models: The proposed model (VAE-GRU, VAE-LSTM) shows faster learning with an average of 7 epochs, while the others tend to take longer to achieve similar results.  

# V. CONCLUSION AND FUTURE WORKS  

We introduced a novel model that reduces dependency on extensive pre-processing by leveraging the VAE latent space for data imputation and recurrent VAE for preserving temporal dynamics.  

Our proposed VAE-RNN model demonstrated superior performance in handling data preprocessing compared to manual techniques and predicting glucose levels compared to traditional RNN-based models (LSTM, Bi-LSTM, GRU, BiGRU). Experimental results showed that VAE-GRU achieved the lowest RMSE and MAPE values for both 30-minute and 1- hour prediction horizons. Clinically, VAE-GRU exhibited the highest accuracy in zones A and B while minimizing critical prediction errors in zones D and E. Despite the competitive performance of Bi-RNN models in quantitative metrics, they were less effective in clinical applications. Additionally, our proposed models showed significant efficiency in training time, converging in fewer epochs compared to other models.  

# Limitations  

The proposed architecture necessitates resampling the data to a uniform frequency, which limits its ability to handle varying time intervals. Moreover, since the model relies on an RNN as a core component, it inherits the common RNN issues of vanishing and exploding gradients.  

Future Works  

Future research will focus on several key areas to further enhance the performance and applicability of our models:  

• Model Generalization: Exploring the generalizability of the proposed models across diverse patient populations and varying clinical settings to ensure robust performance.   
• Real-time Implementation: Developing and testing realtime implementations of the models in clinical environments to evaluate their practical utility and integration into existing healthcare systems.   
• Multi-modal Data Integration: Incorporating additional data sources, such as activity trackers, and dietary logs, to improve prediction accuracy and provide comprehensive patient monitoring.   
• Adaptive Learning: Investigating adaptive learning techniques that allow the models to continuously learn and update from new patient data, enhancing their long-term accuracy and reliability.   
• Explainability and Interpretability: Enhancing the explainability and interpretability of the models to provide healthcare professionals with actionable insights and increase trust in AI-driven decisions.  

# REFERENCES  

[1] Fiorini, N., Ranwez, S., Montmain, J. & Ranwez, V. USI: a fast and accurate approach for conceptual document annotation. BMC Bioinformatics. 16 pp. 83:1-83:10 (2015), http://dx.doi.org/10.1186/s12859-015-0513-4   
[2] Resnik, P. Using Information Content to Evaluate Semantic Similarity in a Taxonomy. Proceedings Of The Fourteenth International Joint Conference On Artificial Intelligence, IJCAI 95, Montr´eal Qu´ebec, Canada, August 20-25 1995, 2 Volumes. pp. 448-453 (1995), http://ijcai.org/Proceedings/95-1/Papers/059.pdf   
[3] Harispe, S., Ranwez, S., Janaqi, S. & Montmain, J. Semantic Similarity from Natural Language and Ontology Analysis. (Morgan Claypool Publishers,2015), http://dx.doi.org/10.2200/S00639ED1V01Y201504HLT027   
[4] Guan, Z., Li, H., Liu, R., Cai, C., Liu, Y., Li, J., Wang, X., Huang, S., Wu, L., Liu, D., Yu, S., Wang, Z., Shu, J., Hou, X., Yang, X., Jia, W. & Bin Sheng Artificial intelligence in diabetes management: Advancements, opportunities, and challenges. Cell Reports Medicine. (2023)   
[5] AZ, W., E, ˚A., S, W., D, A., L, M., T, B. & G., H. Data-driven modeling and prediction of blood glucose dynamics: Machine learning applications in type 1 diabetes.. Artificial Intelligence In Medicine. (2019)   
[6] Zheng, M., Ni, B. & Samantha Kleinberg Automated meal detection from continuous glucose monitor data through simulation and explanation. Journal Of The American Medical Informatics Association. (2019)   
[7] Ma, K., Chen, H. & Lin, S. An Ensemble Learning Approach for Exercise Detection in Type 1 Diabetes Patients. (2023)   
[8] Woldaregay, A., Botsis, E., Albers, D., Lena Mamykina & Gunnar Hartvigsen Data-Driven Blood Glucose Pattern Classification and Anomalies Detection: Machine-Learning Applications in Type 1 Diabetes. Journal Of Medical Internet Research. (2019)   
[9] Cappon, G., Prendin, F., Facchinetti, A., Sparacino, G. & Favero, S. Individualized Models for Glucose Prediction in Type 1 Diabetes: Comparing Black-Box Approaches to a Physiological White-Box One. IEEE Transactions On Bio-medical Engineering. (2023)   
[10] Man, C., Micheletto, F., Lv, D., Breton, M., Boris Kovatchev & Claudio Cobelli The UVA/PADOVA Type 1 Diabetes Simulator: New Features. Journal Of Diabetes Science And Technology. (2014)   
[11] Zarkogianni, K., Mitsis, K., Arredondo, M., Fico, G., Fioravanti, A. & Nikita, K. Neuro-fuzzy based glucose prediction model for patients with Type 1 diabetes mellitus. IEEE-EMBS International Conference On Biomedical And Health Informatics (BHI). pp. 252-255 (2014)   
[12] Allam, F., Nossai, Z., Gomma, H., Ibrahim, I. & Abdelsalam, M. A Recurrent Neural Network Approach for Predicting Glucose Concentration in Type-1 Diabetic Patients. Engineering Applications Of Neural Networks. pp. 254-259 (2011)   
[13] Kingma, D. & Welling, M. Auto-Encoding Variational Bayes. (2022)   
[14] Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A. & Bengio, Y. Generative Adversarial Networks. (2014)   
[15] Chung, J., Kastner, K., Ballas, N., Gomez, S., Schuster, M., Le, Q. & Ng, A. A Recurrent Latent Variable Model for Sequential Data. ArXiv Preprint arXiv:1506.02216. (2015)   
[16] Bai, S., Kolter, J. & Koltun, V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. ArXiv Preprint arXiv:1803.01271. (2018)   
[17] Mulyadi, A., Jun, E. & Suk, H. Uncertainty-Aware Variational-Recurrent Imputation Network for Clinical Time Series. (2020)   
[18] Cruz Castan˜eda, W. & Bertemes Filho, P. Synthetic Health data Generation for Enhancement of Non-Invasive Diabetes AI-Based Prediction. (2023,8)   
[19] Stellwagen, E. & Tashman, L. ARIMA: The Models of Box and Jenkins. Foresight: Int. J. Appl. Forecast.. pp. 28-33 (2013,1)   
[20] Joshi, M., Pant, D., Karn, R., Heikkonen, J. & Kanth, R. Meta-Learning, Fast Adaptation, and Latent Representation for Head Pose Estimation. Proceedings Of The XXth Conference Of Open Innovations Association FRUCT. 31 (2022,4)   
[21] Miller, A., Foti, N. & Fox, E. Learning Insulin-Glucose Dynamics in the Wild. MLHC. (2020), https://arxiv.org/pdf/2008.02852.pdf   
[22] Marling, C. & Bunescu, R. The OhioT1DM dataset for Blood Glucose Level Prediction: Update 2020. CEUR Workshop Proc.. 2675 pp. 71-74 (2020,9)   
[23] Clarke, W., Cox, D., McMahon, M., Hanna, L. & Jeffcoate, S. Evaluating clinical accuracy of systems for self-monitoring of blood glucose. Diabetes Care. 10, 622-628 (1987)   
[24] Tawakuli, A., Havers, B., Gulisano, V., Kaiser, D. & Engel, T. Timeseries data preprocessing: A survey and an empirical analysis. Journal Of Engineering Research. (2024)  