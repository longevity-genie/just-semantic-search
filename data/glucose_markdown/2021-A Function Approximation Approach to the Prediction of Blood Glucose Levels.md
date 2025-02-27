# A function approximation approach to the prediction of blood glucose levels  

H.N. Mhaskar ∗1, S.V. Pereverzyev $^2$ and M.D. van der Walt †3  

$^{1}$ Institute of Mathematical Sciences, Claremont Graduate University, Claremont, CA, 91711 USA   
2The Johann Radon Institute for Computational and Applied Mathematics (RICAM), Austrian Academy of Sciences, Linz, Austria. $^3$ Department of Mathematics and Computer Science, Westmont College, Santa Barbara, CA, USA  

# Abstract  

The problem of real time prediction of blood glucose (BG) levels based on the readings from a continuous glucose monitoring (CGM) device is a problem of great importance in diabetes care, and therefore, has attracted a lot of research in recent years, especially based on machine learning. An accurate prediction with a 30, 60, or 90 minute prediction horizon has the potential of saving millions of dollars in emergency care costs. In this paper, we treat the problem as one of function approximation, where the value of the BG level at time $t+h$ (where $h$ the prediction horizon) is considered to be an unknown function of $d$ readings prior to the time $t$ . This unknown function may be supported in particular on some unknown submanifold of the $d$ -dimensional Euclidean space. While manifold learning is classically done in a semi-supervised setting, where the entire data has to be known in advance, we use recent ideas to achieve an accurate function approximation in a supervised setting; i.e., construct a model for the target function. We use the state-of-the-art clinically relevant PRED-EGA grid to evaluate our results, and demonstrate that for a real life dataset, our method performs better than a standard deep network, especially in hypoglycemic and hyperglycemic regimes. One noteworthy aspect of this work is that the training data and test data may come from different distributions.  

Keywords: supervised learning, Hermite polynomial, continuous glucose monitoring, blood glucose prediction, prediction error-grid analysis.  

# 1 Introduction  

Diabetes is a major disease affecting humanity. According to the 2020 National Diabetes Statistics report [3], more than 34,000,000 people had diabetes in the United States alone in 2018, contributing to nearly 270,000 deaths and costing nearly 327 billion dollars in 2017. There is so far no cure for diabetes, and one of the important ingredients in keeping it in control is to monitor blood glucose levels regularly. Continuous glucose monitoring (CGM) devices are used increasingly for this purpose. These devices typically record blood sugar readings at 5 minute intervals, resulting in a time series. This paper aims at predicting the level and rate of change of the sugar levels at a medium range prediction horizon; i.e., looking ahead for 30-60 minutes. A clinically reliable prediction of this nature is extremely useful in conjunction with other communication devices in avoiding unnecessary costs and dangers. For example, hypoglycemia may lead to loss of consciousness, confusion or seizures [9]. However, since the onset of hypoglycemic periods are often silent (without indicating symptoms), it can be very hard for a human to predict in advance. If these predictions are getting communicated to a hospital by some wearable communication device linked with the CGM device, then a nurse could alert the patient if the sugar levels are expected to reach dangerous levels, so that the patient could drive to the hospital if necessary, avoiding a dangerous situation, not to mention an expensive call for an ambulance.  

Naturally, there are many papers dealing with prediction of blood glucose (BG) based not just on CGM recordings but also other factors such as meals, exercise, etc. In particular, machine learning has been used extensively for this purpose. We refer to [9] for a recent review of BG prediction methods based on machine learning, with specific focus on predicting and detecting hypoglycemia.  

The fundamental problem of machine learning is the following. We have (training) data of the form $\{(\mathbf{x}_{j},f(\mathbf{x}_{j})+$ $\epsilon_{j})\}$ , where, for some integer $d\geq1$ , $\mathbf{x}_{j}\,\in\,\mathbb{R}^{d}$ are realizations of a random variable, $f$ is an unknown real valued function, and the $\epsilon_{j}$ ’s are realizations of a mean zero random variable. The distributions of both the random variables are not known. The problem is to approximate $f$ with a suitable model, especially for the points which are not in the training data. For example, in the prediction of a time series $x_{0},x_{1},\ldots$ with a prediction horizon of $h$ based on $d$ past readings can be (and is in this paper) viewed as a problem of approximating an unknown function $f$ such that $x_{j+h}=f(x_{j-d+1},\ldots,x_{j})$ for each $j$ . Thus, $\mathbf{x}_{j}=(x_{j-d+1},\ldots,x_{j})$ and $y_{j}=x_{j+h}$ .  

Such problems have been studied in approximation theory for more than a century. Unfortunately, classical approximation theory techniques do not apply directly in this context, mainly because the data does not necessarily become dense on any known domain such as a cube, sphere, etc., and we cannot control the locations of the points $\mathbf{{x}}_{j}$ . This is already clear from the example given above – it is impossible to require that the BG readings be at a pre-chosen set of points.  

Manifold learning tries to ameliorate this problem by assuming that the data lies on some unknown manifold, and developing techniques to learn various quantities related to the manifold, in particular, the eigen-decomposition of the Laplace-Beltrami operator. Since these objects must be learnt from the data, these techniques are applicable in the semi-supervised setting, in which we have all the data points $\mathbf{x}_{j}$ available in advance. In [8], we have used deep manifold learning to predict BG levels using CGM data. Being in the semi-supervised setting, this approach cannot be used directly for prediction based on data that was not included in the original data set we worked with.  

Recently, we have discovered [7] a more direct method to approximate functions on unknown manifolds without trying to learn anything about the manifold itself, and giving an approximation with theoretically guaranteed errors on the entire manifold; not just the points in the original data set. The purpose of this paper is to use this new tool for BG prediction based on CGM readings which can be used on patients not in the same data sets, and for readings for the patients in the data set which are not included among the training data.  

In this context, the numerical accuracy of the prediction alone is not clinically relevant. The Clarke Error-Grid Analysis (C-EGA) [2] is devised to predict whether the prediction is reasonably accurate, or erroneous without serious consequences (clinically uncritical), or results in unnecessary treatment (overcorrection), or dangerous (either because it fails to identify a rise or drop in BG or because it confuses a rise in BG for a drop or vice versa). The Prediction Error-Grid Analysis (PRED-EGA), on the other hand, categorizes a prediction as accurate, erroneous with benign (not serious) consequences or erroneous with potentially dangerous consequences in the hypoglycemic (low BG), euglycemic (normal BG), and hyperglycemic (high BG) ranges separately. This stratification is of great importance because consequences caused by a prediction error in the hypoglycemic range are very different from ones in the euglycemic or the hyperglycemic range. In addition, the categorization is based not only on reference BG estimates paired with the BG estimates predicted for the same moments (as C-EGA does), but also on reference BG directions and rates of change paired with the corresponding estimates predicted for the same moments. It is argued in [14] that this is a much better clinically meaningful assessment tool than C-EGA, and is therefore the assessment tool we choose to use in this paper.  

Since different papers on the subject use different criterion for assessment it is impossible for us to compare our results with those in most of these papers. Besides, a major objective of this paper is to illustrate the use and utility of the theory in [7] from the point of view of machine learning. So, we will demonstrate the superiority of our results in the hypoglycemic and hyperglycemic regimes over those obtained by a blind training of a deep network using the MATLAB Deep Learning Toolbox. We will also compare our results with those obtained by the techniques used in [10, 6, 8]. However, we note that these methods are not directly comparable to ours. In [10], the authors use the data for only one patient as training data, which leads to a meta-learning model that can be used serendipitously on other patients, although a little bit of further training is still required for each patient to apply this model. In contrast, we require a more extensive “training data”, but there is no training in the classical sense, and the application to new patients consists of a simple matrix vector multiplication. The method in [6] is a linear extrapolation method, where the coefifcients are learnt separately on each patient. It does not result in a model that can be trained once for all and applied to other patients in a simple manner. As pointed out earlier, the method in [8] requires that the entire data set be available even before the method can start to work. So, unlike our method, it cannot be trained on a part of the data set and applied to a completely different data set, not available at the beginning.  

We will explain the problem and the evaluation criterion in Section 3. Prior work on this problem is reviewed briefly in Section 2. The methodology and algorithm used in this paper are described in Section 4. The results are discussed in Section 5. The mathematical background behind the methods described in Section 4 is summarized in Appendix A.  

# 2 Prior work  

Given the importance of the problem, many researchers have worked on it in several directions. Below we highlight the main trends relevant to our work.  

The majority of machine learning models (30% of those studied in the survey paper [9]) are based on artificial neural networks such as convolutional neural networks, recurrent neural networks and deep learning techniques. For example, in [17], the authors develop a deep learning model based on a dilated recurrent neural network to achieve 30-minute blood glucose prediction. The authors assess the accuracy of their method by calculating the root mean squared error (RMSE) and mean absolute relative difference (MARD). While they achieve good results, a limitation of the method is the reliance on various input parameters such as meal intake and insulin dose which are often recorded manually and might therefore be inaccurate. In [11], on the other hand, a feed-forward neural network is designed with eleven neurons in the input layer (corresponding to variables such as CGM data, the rate of change of glucose levels, meal intake and insulin dosage), and nine neurons with hyperbolic tangent transfer function in the hidden layer. The network was trained with the use of data from 17 patients and tested on data from 10 other patients for a 75-minute prediction, and evaluated using C-EGA. Although good results are achieved in the C-EGA grid in this paper, a limitation of the method is again the large amount of additional input information necessary to design the model, as described above.  

A second BG prediction class employs decision trees. The authors of [13], for example, use random forests to perform a 30-minute prediction to specifically detect postprandial hypoglycemia, that is, hypoglycemia occurring after a meal. They use statistical analyses like sensitivity and specificity to evaluate their method. In [16], prediction is again achieved by random forests and incorporating a wide variety of additional input features such as physiological and physical activity parameters. Mean Absolute Percentage Error (MAPE) is used as assessment methodology. The authors of [5] use a classification and regression tree (CART) to perform a 15-minute prediction using BG data only, again using sensitivity and specificity to test the performance of their method.  

A third class of methods employ kernel-based regularization techniques to achieve prediction (for example, [10] and references therein), where Tikhonov regularization is used to find the best least square fit to the data, assuming the minimizer belongs to a reproducing kernel Hilbert space (RKHS). Of course, these methods are quite sensitive to the choice of kernel and regularization parameters. Therefore, the authors in [10] develop methods to choose both the kernel and regularization parameter adaptively, or through meta-learning (“learning to learn”) approaches. C-EGA and PRED-EGA are used as performance metrics.  

Time series techniques are also employed in the literature. In [12], a tenth-order auto-regression (AR) model is developed, where the AR coefifcients are determined through a regularized least square method. The model is trained patient-by-patient, typically using the first 30% of the patient’s BG measurements, for a 30-minute or 60-minute prediction. The method is tested on a time series containing glucose values measured every minute, and evaluation is again done through the C-EGA grid. The authors in [15] develop a first-order AR model, patient-bypatient, with time-varying AR coefifcients determined through weighted least squares. Their method is tested on a time series containing glucose values measured every three minutes, and quantified using statistical metrics such as RMSE. As noted in [10], these methods seem to be sensitive to gaps in the input data.  

A summary of the methods described above is given in Table 1.  

It should be noted that the majority of BG prediction models (63.64% of those analyzed in [9]), including many of the references described above, are dependent on additional input features (such as meal intake, insulin dosage and physical activity) in addition to historical BG data. Since such data could be inaccurate and hard to obtain, we have intentionally made the design decision in our work to base our method on historical BG data only.  

Table 1 A summary of related work on BG prediction. $\mathrm{PH}=$ prediction horizon, DL $=$ deep learning, NN $=$ neural network, RF $=$ random forest, $\mathrm{CART}\,=$ classification and regresssion tree, KBR $=$ kernel-based regularization, AR $=$ autoregression. RMSE $=$ root mean squared error, $\mathrm{MARD}=$ mean absolute relative difference, C-EGA $=$ Clarke error-grid analysis, MAPE $=$ mean absolute percentage error, PRED-EGA $=$ prediction error-grid analysis.   

![](images/9d7da2f6ffd7a629d71dcfc9df765513b6e9c3a2cb5b45f1d2e98672f093901f.jpg)  

# 3 The set-up  

We use two different clinical data sets provided by the DirectNet Central Laboratory [1], which lists BG levels of different patients taken at 5-minute intervals with the CGM device; i.e., for each patient $p$ in the patient set $P$ , we are given a time series $\{s_{p}(t_{j})\}$ , where $s_{p}(t_{j})$ denotes the BG level at time $t_{j}$ . The relevant details of the two data sets are described in Table 2.  

Table 2 Summary statistics for datasets D and J   

![](images/468f26345e29fdbb779d8c5a39de99956f3302ca2740c0a2ac126f430f6ba757.jpg)  

Our goal is to predict for each $j$ , the level $s_{p}(t_{j+h})$ , given readings $s_{p}(t_{j}),\ldots,s_{p}(t_{j-d+1})$ for appropriate values of $h$ and $d$ . We took $d=7$ (a sampling horizon $t_{j}-t_{j-d+1}$ of 30 minutes has been suggested as the optimal one for BG prediction in [6, 4]). We tested prediction horizons of 30 minutes ( $h=6$ ) (the most common prediction horizon according to the analysis in [9]), 60 minutes ( $h=12$ ) and 90 minutes ( $h=18$ ).  

# 4 Methodology in the current paper  

# 4.1 Data reformulation  

For each patient in the set $P$ of patients in the data sets, the data is in the form of a time series $\{s_{p}(t_{j})\}_{j=1}^{N_{p}}$ of $p$ BG levels at time $t_{j}$ , where $t_{j}-t_{j-1}=5$ minutes. We re-organize the data in the form  

$$
\mathcal{P}^{*}=\left\{\left(\left(s_{p}(t_{j-d+1}),\cdot\cdot\cdot\,,s_{p}(t_{j})\right),s_{p}(t_{j+h})\right):j=d,\cdot\cdot\cdot\,,N_{p}-h,\ p\in P\right\}.
$$  

For any patient $p\in P$ , we will abbreviate  

$$
\mathbf{x}_{p,j}=(s_{p}(t_{j-d+1}),\cdot\cdot\cdot\,,s_{p}(t_{j}))\,,\qquad y_{p,j}=s_{p}(t_{j+h}),
$$  

and write  

$$
j-d+1),\cdot\cdot\cdot,s_{p}(t_{j})):j=d,\cdot\cdot\cdot,N_{p}-h,\ p\in P\}
$$  

The problem of BG prediction is then seen as the classical problem of machine learning; i.e., to find a functional relationship $f$ such that $f(\mathbf{x}_{p,j})\approx y_{p,j}$ , for all $p$ and $j=d,\cdot\cdot\cdot\,,N_{p}-h$ .  

# 4.2 Training data selection  

To obtain training data, we form the training set $C$ by randomly selecting (according to a uniform probability distribution) $c\%$ of the patients in $P$ . The training data are now defined to be all the data of each patient in $C$ , that is,  

$$
\mathcal{C}^{\star}:=\left\{(\mathbf{x}_{p,j},y_{p,j})=\big((s_{p}(t_{j-d+1}),\cdot\cdot\cdot\cdot,s_{p}(t_{j})\big)\,,s_{p}(t_{j+h})\big):j=d,\cdot\cdot\cdot,N_{p}-h,\ p\in C\right\}.
$$  

Since we define the training and test data in terms of patients, choosing the entire data for a patient in the training (respectively, test) data, we may now simplify the notation by writing $(\mathbf{x}_{j},y_{j})$ rather than $(\mathbf{x}_{p,j},y_{p,j})$ , with the understanding that if $\mathbf{x}_{j}$ represents a data point for a patient $p$ , $y_{j}$ is the corresponding value for the same patient $p$ . For future reference, we also define  

$$
{\mathcal{C}}:=\{\mathbf{x}_{j}=(s_{p}(t_{j-d+1}),\cdot\cdot\cdot\mathbf{\mu},s_{p}(t_{j})):p\in C\}
$$  

and the testing patient set $Q=P\setminus C$ , with  

$$
\mathcal{Q}:=\mathcal{P}\setminus\mathcal{C}=\left\{\mathbf{x}_{j}=\left(s_{p}(t_{j-d+1}),\cdot\cdot\cdot\cdot,s_{p}(t_{j})\right):p\in Q\right\}.
$$  

# 4.3 BG range classification  

To get accurate results in each BG range, we want to be able to adapt the training data and parameters used for predictions in each range – as noted previously, consequences of prediction errors in the separate BG ranges are very different. To this end, we divide the measurements in $\mathcal{C}$ into three clusters $C_{o},C_{e}$ and $\ensuremath{{\mathcal{C}}}_{r}$ , based on , the BG $y_{j}$ value at time $t_{j+h}$ , for each $\mathbf{\boldsymbol{x}}_{j}$ :  

$$
\begin{array}{r}{\mathcal{C}_{o}=\{\mathbf{x}_{j}\in\mathcal{C}:0\leq y_{j}\leq70\}\ \mathrm{(hypoglycemia)};\quad}\\ {\mathcal{C}_{e}=\{\mathbf{x}_{j}\in\mathcal{C}:70<y_{j}\leq180\}\ \mathrm{(euglycemia)};\quad}\\ {\mathcal{C}_{r}=\{\mathbf{x}_{j}\in\mathcal{C}:180<y_{j}\leq450\}\ \mathrm{(hyperglycemia)},}\end{array}
$$  

with  

$$
{\mathcal C}_{\ell}^{\star}=\left\{({\bf x}_{j},y_{j}):{\bf x}_{j}\in{\mathcal C}_{\ell}\right\},\quad\ell\in\left\{o,e,r\right\}.
$$  

In addition, we also classify each $\mathbf{x}_{j}\in\mathcal{Q}$ as follows:  

$$
\begin{array}{r}{Q_{o}=\{\mathbf{x}_{j}\in Q:0\leq x_{j,d}\leq70\}\ \mathrm{(hypoglycemia)};}\\ {Q_{e}=\{\mathbf{x}_{j}\in Q:70<x_{j,d}\leq180\}\ \mathrm{(euglycemia)};}\\ {Q_{r}=\{\mathbf{x}_{j}\in Q:180<x_{j,d}\leq450\}\ \mathrm{(hyperglycemia)}.}\end{array}
$$  

Ideally, the classification of each $\mathbf{x}_{j}\,\in\,\mathcal{Q}$ should also be done based on the value of $y_{j}$ (as was done for the classification of the training data $\mathcal{C}$ ). However, since the values $y_{j}$ are only available for the training data and not for the test data, we use the next best available measurement for each $\mathbf{\boldsymbol{x}}_{j}$ , namely $x_{j,d}$ , the BG value at time $t_{j}$ .  

# 4.4 Prediction  

Before making the BG predictions, it is necessary to scale the input readings so that the components of each $\mathbf{x}_{j}\in\mathcal{P}$ are in $[-1/2,1/2]$ . This is done through the transformation  

$$
\mathbf{x}_{j}\mapsto\frac{2\mathbf{x}_{j}-(M+m)}{2(M-m)},
$$  

With $\hat{F}_{n,\alpha}\,=\,\hat{F}_{n,\alpha}(Y,{\bf x}_{j})$ defined as in (A.3) used with training data $Y$ and evaluated at a point $\mathbf{{x}}_{j}$ , we are finally ready to compute the BG prediction for each $\mathbf{x}_{j}\in\mathcal{Q}$ by  

$$
f(\mathbf{x}_{j}):=\left\{\hat{F}_{n_{o},\alpha_{o}}(\mathcal{C}_{o}^{\star},\mathbf{x}_{j}),\quad\mathbf{x}_{j}\in\mathcal{Q}_{o},\right.
$$  

The construction of the estimator $\hat{F}_{n,\alpha}$ involves several technical details regarding the classical Hermite polynomials.   
For ease of reading, the details of this construction are deferred to the appendix.  

# 4.5 Evaluation  

To evaluate the performance of the final output $f(\mathbf{x}_{j})$ , $\mathbf{x}_{j}\in\mathcal{Q}$ , we use the PRED-EGA mentioned in Section 3. Specifically, a PRED-EGA grid is constructed by using comparisons of $f(\mathbf{x}_{j})$ with the reference value $y_{j}$ . This involves comparing  

$$
f(\mathbf{x}_{j})\quad{\mathrm{with}}\quad y_{j}
$$  

as well as the rates of change  

$$
\frac{f(\mathbf{x}_{j+1})-f(\mathbf{x}_{j-1})}{2(t_{j+1}-t_{j-1})}\quad\mathrm{with}\quad\frac{y_{j+1}-y_{j-1}}{2(t_{j+1}-t_{j-1})},
$$  

for all $\mathbf{x}_{j}\in\mathcal{Q}$ . Based on these comparisons, PRED-EGA classifies $f(\mathbf{x}_{j})$ as being Accurate, Benign or Erroneous.  

We repeat the entire process described in Subsections 4.2 - 4.5 for a fixed number of trials, after which we report the average of the PRED-EGA grid placements, over all $\mathbf{x}_{j}\in\mathcal{Q}$ and over all trials, as the final evaluation.  

A summary of the method is given in Algorithm 1.  

# 5 Results and discussion  

As mentioned in Section 3, we apply our method to data provided by the DirecNet Central Laboratory. Time series for the 25 patients in data set D and the 25 patients in data set J that contain the largest number of BG measurements are considered. These specific data sets were obtained to study the performance of CGM devices in children with Type I diabetes; as such, all of the patients are less than 18 years old. Summary statistics of the two data sets are provided in Table 2. Our method is a general purpose algorithm, where these details do not play any significant role, except in affecting the outcome of the experiments.  

We provide results obtained by implementing our method in MATLAB, as described in Algorithm 1. For our implementation, we employ a sampling horizon $t_{j}-t_{j-d+1}$ of 30 minutes ( $d\,=\,7$ ), 50% training data ( $c\,=\,50$ ) (which is comparable to approaches followed in for example [12]) and a total of 100 trials ( $T=100$ ). For all the experiments, the function approximation parameters $\pi_{o},n_{e}$ and $n_{r}$ referenced in Algorithm 1 were chosen in the range $\{3,\ldots,7\}$ with $\alpha_{o}=\alpha_{e}=\alpha_{r}=1$ . We provide results for prediction horizons $t_{j+m}-t_{j}$ of 30 minutes ( $h=6$ ), 60 minutes ( $h=12$ ) and 90 minutes ( $h=18$ ) (again, comparable to prediction horizons in for example [12, 13, 17]). After testing our method on all 25 patients in each data set separately, the average PRED-EGA scores (in percent) are displayed in Tables 3 and 4.  

For comparison, Tables 3 and 4 also display predictions on the same data sets using four other methods:  

(i) MATLAB’s built-in Deep Learning Toolbox. We used a network architecture consisting of three fully connected convolutional and output layers and appropriate input, normalization, activation, and averaging layers. As before, we used 50% training data and averaged the experiment over 100 trials, for prediction windows of 30 minutes, 60 minutes and 90 minutes and data sets D and J.  

(ii) The diffusion geometry approach, based on the eigen-decomposition of the Laplace-Beltrami operator, we followed in our 2017 paper [8]. As mentioned in Section 1, this approach can be classified as semi-supervised learning, where we need to have all the data points $\mathbf{\boldsymbol{x}}_{j}$ available in advance. Because of this requirement,  

# Algorithm 1 Deep Network for BG prediction  

input : Time series $\{s_{p}(t_{j})\}$ , $p\in P$ , formatted as $\mathcal{P}=\{\mathbf{x}_{j}\}$ with $\mathbf{x}_{j}=(s_{p}(t_{j-d+1}),\cdot\cdot\cdot\,,s_{p}(t_{j}))$ and $y_{j}=s_{p}(t_{j+h})$ $d\in\mathbb{N}$ (specifies sampling horizon), $h\in\mathbb N$ (specifies prediction horizon) $c\in(0,100)$ (percentage of data used for training) $n_{o},n_{e},n_{r}\in\mathbb{N}$ , $\alpha_{o},\alpha_{e},\alpha_{r}\in(0,1]$ (parameters for function approximation) Let $C$ contain $c\%$ of patients from $P$ (drawn according to uniform prob. distr.) Set $\mathcal{C}=\{\mathbf{x}_{j}\}$ and $\mathcal{C}^{\star}=\{(\mathbf{x}_{j},y_{j})\}$ for all patients $p\in C$ Set ${\mathcal{Q}}={\mathcal{P}}\setminus{\mathcal{C}}$  

output: Prediction $f(\mathbf{x}_{j})\approx s_{p}(t_{j+h})$ for $\mathbf{x}_{j}\in\mathcal{Q}$ . Set $\mathcal{C}_{o}=\{\mathbf{x}_{j}\in\mathcal{C}:0\leq y_{j}\leq70\}$ Set $\mathcal{C}_{e}=\{\mathbf{x}_{j}\in\mathcal{C}:70<y_{j}\leq180\}$ Set $\mathcal{C}_{r}=\{\mathbf{x}_{j}\in\mathcal{C}:180<y_{j}\leq450\}$ Set $\begin{array}{r}{\mathcal{C}_{\ell}^{\star}=\left\{(\mathbf{x}_{j},y_{j}):\mathbf{x}_{j}\in\mathcal{C}_{\ell}\right\},}\end{array}$ ℓ∈{o, e, r}  

Set $\mathcal{Q}_{o}=\{\mathbf{x}_{j}\in\mathcal{Q}:0\leq x_{j,d}\leq70\}$ Set $\mathcal{Q}_{e}=\{\mathbf{x}_{j}\in\mathcal{Q}:70<x_{j,d}\leq180\}$ Set $\mathcal{Q}_{r}=\left\{\mathbf{x}_{j}\in\mathcal{Q}:180<x_{j,d}\leq450\right\}$  

Set $M=\operatorname*{max}\{x_{j,k}:\mathbf{x}_{j}\in\mathcal{C}\}$ and $m=\operatorname*{min}\{x_{j,k}:\mathbf{x}_{j}\in\mathcal{C}\}$ R $\mathbf{x}_{j}\in\mathcal{P}$ $\begin{array}{r}{\mathbf{x}_{j}=\frac{2\mathbf{x}_{j}-(M+m)}{2(M-m)}}\end{array}$  

# end  

for $\ell\in\{o,e,r\}$ do for $\mathbf{x}_{j}\in\mathcal{Q}_{\ell}$ do Compute $f(\mathbf{x}_{j})=\hat{F}_{n_{\ell},\alpha_{\ell}}(\mathcal{C}_{\ell}^{\star},\mathbf{x}_{j})$ end  

end  

this approach cannot be used for prediction based on data that was not included in the original data set we worked with. Indeed, this is one of the main motivations for our current method, which is not dependent on this requirement. Nevertheless, we include a comparison of the diffusion geometry approach with our current method for greater justification of the accuracy of our current method. Again, we used 50% training data and averaged the experiment over 100 trials, for prediction windows of 30 minutes, 60 minutes and 90 minutes and data sets D and J.  

(iii) The Legendre polynomial prediction method followed in our 2013 paper [6]. In this context, the main mathematical problem can be summarized as that of estimating the derivative of a function at the end point of an interval, based on measurements of the function in the past. An important difference between our current method and the Legendre polynoimal approach is that with the latter, learning is based on the data for each patient separately (each patient forms a data set in itself, as explained in Section 1). Therefore, the method is philosophically different from ours. We nevertheless include a comparison for prediction windows of 30 minutes, 60 minutes and 90 minutes and data sets D and J.   
(iv) The Fully Adaptive Regularized Learning (FARL) approach in the 2012 paper [10], which uses meta-learning to choose the kernel and regularization parameters adaptively (briefly described in Section 2). As explained in [10], a strong suit of FARL is that it is expected to be portable from patient to patient without any readjustment. Therefore, the results reported below were obtained by implementing the code and parameters used in [10] directly on data sets D and J for prediction windows of 30 minutes, 60 minutes and 90 minutes, with no adjustments for the kernel and regularization parameters. As explained earlier, this method is also not directly compared to our theoretically well founded method.  

The percentage accurate predictions and predictions with benign consequences in all three BG ranges, as obtained using all five methods, for a 30 minute, 60 minute and 90 minute prediction window, are displayed visually in Figures 1-2.  

By comparing the percentages in Tables 3-4 and the bar heights in Figures 1-2, it is clear that our method far outperforms the competitors in the hypoglycemic range, except for the 30 minute prediction on data set D, where the diffusion geometry approach is slightly more accurate – but it is important to remember that the diffusion geometry approach has all the data points available from the outset. The starkest difference is between our method and Matlab’s Deep Learning Toolbox – the latter achieves less than $1.5\%$ accurate and benign consequence predictions regardless of the size of the prediction window or data set. There is also a marked difference between our method and the Legendre polynomial approach, especially for longer prediction horizons. Our method achieves comparable or higher accuracy in the hyperglycemic range, except for the 60 and 90 minute predictions on data set J, where the diffusion geometry and FARL approaches achieve more accurate predictions. The methods display comparable accuracy in the euglycemic range, except for the 60 and 90 minute predictions on dataset J, where Matlab’s toolbox, the diffusion geometry approach and the FARL method achieve higher accuracy.  

Tables 5-6 display the results when testing those methods that are dependent on training data selection (that is, our current method, the Deep Learning Toolbox and the diffusion geometry approach in our 2017 paper) on different training set sizes (namely, 30%, 50%, 70% and $90\%$ ). For these experiments, we used a fixed prediction window of 30 minutes, and training data was drawn randomly according to a uniform probability distribution in each case. While the Deep Learning Toolbox and diffusion geometry approach sometimes perform slightly better than our method in the euglycemic and hypoglycemic ranges, respectively, our method consistently outperforms both of these in the other two blood glucose ranges. The results are displayed visually as well in Figures 3-4.  

As mentioned in Section 1, a feature of our method is that it does not require any information about the data manifold other than its dimension. So, in principle, one could “train” on one data set and apply the resulting model to another data set taken from a different manifold. We illustrate this by constructing our model based on one of the data sets D or J and testing it on the other data set. Building on this idea, we also demonstrate the accuracy of our method as compared to MATLAB’s Deep Learning Toolbox when trained on data set D and applied to perform prediction on data set J, and vice versa, which is the only other method that is directly comparable with the philosophy behind our method. These results are reported in Tables 7 and 8. It is clear that we are successful in our goal of training on one data set and predicting on a different data set. The percentage accurate predictions and predictions with benign consequences when training and testing on different datasets are comparable to the cases when we are training and testing on the same dataset; in fact, we obtain even better results in the hypoglycemic  

Table 3 Average PRED-EGA scores (in percent) for different prediction horizons on dataset D.   

![](images/2b4402c36ee468a19cd56b059394949447d310a036df8f1194b32fdedc331466.jpg)  

BG range when training and testing across different datasets.  

Figures 5 and 6 display box plots for the 100 trials for the methods dependent on training data selection (that is, our current method, the Deep Learning Toolbox and the 2017 diffusion geometry approach) and prediction windows using data sets D and J, respectively. Our method performs fairly consistently over different trials.  

# 6 Conclusions  

In this paper, we demonstrate a direct method to approximate functions on unknown manifolds [7] in the context of BG prediction for diabetes management. The results are evaluated using the state-of-the-art PRED-EGA grid [14]. Unlike classical manifold learning, our method yields a model for the BG prediction on data not seen during training, even for test data which is drawn from a different distribution from the training data. Our method outperforms other methods in the literature in 30-minute, 60-minute and 90-minute predictions in the hypoglycemic and hyperglycemic BG ranges, and is especially useful for BG prediction for patients that are not included in the training set.  

# Acknowledgments  

The research of HNM is supported in part by NSF grant DMS 2012355 and ARO grant W911NF2110218. The research by SVP has been partly supported by BMK, BMDW, and the Province of Upper Austria in the frame of the COMET Programme managed by FFG in the COMET Module S3AI.  

# References  

[1] DirecNet Central Laboratory. http://direcnet.jaeb.org/Studies.aspx, 2005.  

[2] W.L. Clarke, D. Cox, L.A. Gonder-Frederick, W. Carter, and S.L. Pohl. Evaluating clinical accuracy of systems for self-monitoring of blood glucose. Diabetes care, 10(5):622–628, 1987.  

Table 4 Average PRED-EGA scores (in percent) for different prediction horizons on dataset J.   

![](images/ff523d1014792d4d177345421445dcc2f1d3879f068b32bea58525815f11f143.jpg)  

Table 5 Percentage accurate predictions for different training set sizes and a $30\,\mathrm{\min}$ prediction horizon on dataset D.   

![](images/90424755ae3fce98f6db0025084d9b04359e985b8c27dd906280e207db4a410c.jpg)  

![](images/14aeeb193b2080cd62daf6d30d6a74a5d3f6a6e6b53737af292bc492432411d3.jpg)  
Figure 1 Percentage accurate predictions and predictions with benign consequences in all three BG ranges (blue: hypoglycemia; orange: euglycemia; grey: hyperglycemia) for dataset $\mathbf{D}$ , for all five methods for different prediction windows (30 minutes, 60 minutes and 90 minutes).  

![](images/59dfd8160b43447f2e042d3eb62a9ffc1009f431a576c930ecb15eed5b54d12f.jpg)  
Figure 2 Percentage accurate predictions and predictions with benign consequences in all three BG ranges (blue: hypoglycemia; orange: euglycemia; grey: hyperglycemia) for dataset J, for all five methods for different prediction windows (30 minutes, 60 minutes and 90 minutes).  

Table 6 Percentage accurate predictions for different training set sizes and a $30\,\mathrm{\min}$ prediction horizon on dataset J.   

![](images/29bb473d0a5bd2cece8bda993f0414a9ddabf341a75f74b7eff9466ac6cd728b.jpg)  

![](images/dc1e32d09759096c00c2da0f171dbbeec7f26038073aaba4869cf24cac58a54b.jpg)  
Figure 3 Percentage accurate predictions and predictions with benign consequences in all three BG ranges (blue: hypoglycemia; orange: euglycemia; grey: hyperglycemia) with a 30 minute prediction window on dataset D, for different training set sizes ( $30\%$ , $50\%$ , $70\%$ and $90\%$ ).  

![](images/f9e0a488135ef78535e202e191dc39d180f948f8eec0dca7349c00d69a64978d.jpg)  
Figure 4 Percentage accurate predictions and predictions with benign consequences in all three BG ranges (blue: hypoglycemia; orange: euglycemia; grey: hyperglycemia) with a 30 minute prediction window on dataset J, for different training set sizes $30\%$ , $50\%$ , $70\%$ and $90\%$ ).  

![](images/7934aed09ed4ead6c93d0902ac100d079f937fd4c61d232a06cfafffe1961912.jpg)  
Table 7 Average PRED-EGA scores (in percent) for different prediction horizons on dataset $\mathbf{D}$ , with training data selected from dataset J.  

Table 8 Average PRED-EGA scores (in percent) for different prediction horizons on dataset $\mathbf{J}$ , with training data selected from dataset D.   

![](images/da7dc02b84b74c66d36197708c543fefed867a24d86e066a246d0831540e76d4.jpg)  

![](images/b6b1393ec7f4351f8617d30e2fdd2fee458d362659fe86366c985c9c9143481d.jpg)  
Figure 5 Boxplot for the 100 experiments conducted with $50\%$ training data for each prediction method $\mathrm{\Theta}=$ our method, $\mathrm{{NI}=}$ MATLAB Deep Learning Toolbox, $\textrm{D}=$ diffusion geometry approach) with a 30 minute (top), 60 minute (middle) and 90 minute (bottom) prediction horizon and dataset $\mathbf{D}$ . Each of the graphs show the percentage accurate predictions in the hypoglycemic range (left), euglycemic range (middle) and hyperglycemic range (right).  

[3] Centers for Disease Control, Prevention, et al. National diabetes statistics report, 2020. Atlanta, GA: Centers for Disease Control and Prevention, US Department of Health and Human Services, pages 12–15, 2020. [4] A.C. Hayes, J.J. Mastrototaro, S.B. Moberg, J.C. Mueller Jr, H.B. Clark, M.C.V. Tolle, G.L. Williams, B. Wu, and G.M. Steil. Algorithm sensor augmented bolus estimator for semi-closed loop infusion system, June 16 2009. US Patent 7,547,281. [5] Miyeon Jung, You-Bin Lee, Sang-Man Jin, and Sung-Min Park. Prediction of daytime hypoglycemic events using continuous glucose monitoring data and classification technique. arXiv preprint arXiv:1704.08769, 2017. [6] H.N. Mhaskar, V. Naumova, and S.V. Pereverzyev. Filtered Legendre expansion method for numerical differentiation at the boundary point with application to blood glucose predictions. Applied Mathematics and Computation, 224:835–847, 2013.   
[7] Hrushikesh Mhaskar. Deep gaussian networks for function approximation on data defined manifolds. arXiv preprint arXiv:1908.00156, 2019.   
[8] Hrushikesh N Mhaskar, Sergei V Pereverzyev, and Maria D van der Walt. A deep learning approach to diabetic blood glucose prediction. Frontiers in Applied Mathematics and Statistics, 3:14, 2017. [9] Omer Mujahid, Ivan Contreras, and Josep Vehi. Machine learning techniques for hypoglycemia prediction: Trends and challenges. Sensors, 21(2):546, 2021.   
[10] V. Naumova, S.V. Pereverzyev, and S. Sivananthan. A meta-learning approach to the regularized learning - case study: blood glucose prediction. Neural Networks, 33:181–193, 2012.   
[11] S.M. Pappada, B.D. Cameron, P.M. Rosman, R.E. Bourey, T.J. Papadimos, W. Olorunto, and M.J. Borst. Neural network-based real-time prediction of glucose in patients with insulin-dependent diabetes. Diabetes technology & therapeutics, 13(2):135–141, 2011.   
[12] J. Reifman, S. Rajaraman, A. Gribok, and W.K. Ward. Predictive monitoring for improved management of glucose levels. Journal of Diabetes Science and Technology, 1(4):478–486, 2007.   
[13] Wonju Seo, You-Bin Lee, Seunghyun Lee, Sang-Man Jin, and Sung-Min Park. A machine-learning approach to predict postprandial hypoglycemia. BMC medical informatics and decision making, 19(1):1–13, 2019.   
[14] S. Sivananthan, V. Naumova, C.D. Man, A. Facchinetti, E. Renard, C. Cobelli, and S.V. Pereverzyev. Assessment of blood glucose predictors: the prediction-error grid analysis. Diabetes technology & therapeutics, 13(8):787–796, 2011.   
[15] G. Sparacino, F. Zanderigo, S. Corazza, A. Maran, A. Facchinetti, and C. Cobelli. Glucose concentration can be predicted ahead in time from continuous glucose monitoring sensor time-series. IEEE Transactions on Biomedical Engineering, 54(5):931–937, 2007.   
[16] Mohammad Reza Vahedi, Koenrad B MacBride, Woo Wunsik, Yosep Kim, Chris Fong, Andrew J Padilla, Mohammad Pourhomayoun, Alex Zhong, Sameer Kulkarni, Siddharth Arunachalam, et al. Predicting glucose levels in patients with type1 diabetes based on physiological and activity data. In Proceedings of the 8th ACM MobiHoc 2018 Workshop on Pervasive Wireless Healthcare Workshop, pages 1–5, 2018.   
[17] Taiyu Zhu, Kezhi Li, Jianwei Chen, Pau Herrero, and Pantelis Georgiou. Dilated recurrent neural networks  

![](images/c62278cd020904fc59535fcef84b6886d12e1e3e04db871c61d31429b3e04f7b.jpg)  
Figure 6 Boxplot for the 100 experiments conducted with $50\%$ training data for each prediction method $\mathrm{\Theta}=$ our method, $\mathrm{{NI}=}$ MATLAB Deep Learning Toolbox, $\textrm{D}=$ diffusion geometry approach) with a 30 minute (top), 60 minute (middle) and 90 minute (bottom) prediction horizon and dataset J. Each of the graphs show the percentage accurate predictions in the hypoglycemic range (left), euglycemic range (middle) and hyperglycemic range (right).  

for glucose forecasting in type 1 diabetes. Journal of Healthcare Informatics Research, 4(3):308–324, 2020.  

# A Theoretical aspects  

In this section, we describe the theoretical background behind the estimator $\widehat{F}_{n,\alpha}({\cal T},\circ)$ used in this paper to predict the BG level. We focus only on the essential definitions, referring to [7] for details.  

Let $d\geq q\geq1$ be integers, $\mathbb{X}$ be a $q$ dimensional, compact, connected, sub-manifold of $\mathbb{R}^{d}$ (without boundary), with geodesic distance $\rho$ and volume measure $\mu^{*}$ , normalized so that $\mu^{*}(\mathbb{X})=1$ . The operator $\hat{F}_{n,\alpha}$ does not require  

any knowledge of the manifold other than $q$ . Our construction is based on the classical Hermite polynomials which are best introduced by the recurrence relations (A.1). The orthonormalized Hermite polynomial $h_{k}$ of degree $k$ is defined recursively by  

$$
\begin{array}{l}{{h_{k}(x):=\sqrt{\frac{2}{k}}x h_{k-1}(x)-\sqrt{\frac{k-1}{k}}h_{k-2}(x),\qquad k=2,3,\cdots,}}\\ {{h_{0}(x):=\pi^{-1/4},\,\,h_{1}(x):=\sqrt{2}\pi^{-1/4}x.}}\end{array}
$$  

We write  

$$
\psi_{k}(x):=h_{k}(x)\exp(-x^{2}/2),\qquad x\in\mathbb{R},\ k\in\mathbb{Z}_{+}.
$$  

The functions $\{\psi_{k}\}_{k=0}^{\infty}$ are an orthonormal set with respect to the Lebesgue measure:  

$$
\int_{\mathbb{R}}\psi_{k}(x)\psi_{j}(x)d x={\left\{1,\begin{array}{l l}{*}&{{\mathrm{if~}}k=j,}\\ {0,}&{{\mathrm{~otherwise.}}}\end{array}\right.}
$$  

In the sequel, we fix an infinitely differentiable function $H:[0,\infty)\to[0,1]$ , such that $H(t)=1$ if $0\le t\le1/2$ , and $H(t)=0$ if $t\geq1$ . We define for $x\in\mathbb R$ , $m\in\mathbb{Z}_{+}$ :  

$$
\begin{array}{r}{\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\!\dot{m},q(x):=\left\{\displaystyle\!\!\!\!\!\left(\frac{\pi^{-1/4}(-1)^{m}\frac{\sqrt{(2m)!}}{2^{m}m!}\psi_{2m}(x),}{\frac{1}{7^{(2q-1)/4}T((q-1)/2)}\displaystyle\sum_{\ell=0}^{m}(-1)^{\ell}\frac{T((q-1)/2+m-\ell)}{(m-\ell)!}\displaystyle\frac{\sqrt{(2\ell)!}}{2^{\ell}\ell!}\psi_{2\ell}(x),}\right.\quad\mathrm{if~}q\geq2,}\end{array}
$$  

and the kernel $\widetilde{\Phi}_{n,q}$ for $x\in\mathbb R$ , $n\in\mathbb{Z}_{+}$ by  

$$
\widetilde{\phi}_{n,q}(x):=\sum_{m=0}^{\lfloor n^{2}/2\rfloor}H\left(\frac{\sqrt{2m}}{n}\right)\mathcal{P}_{m,q}(x).
$$  

We assume here a noisy data of the form $(\mathbf{y},\epsilon)$ , with a joint probability distribution $\tau$ and assume further that the marginal distribution of y with respect to $\tau$ has the form $d\nu^{*}=f_{0}d\mu^{*}$ for some $f_{0}\in C(\mathbb{X})$ . In place of $f(\mathbf{y})$ , we consider a noisy variant $\mathcal{F}(\mathbf{y},\epsilon)$ , and denote  

$$
f(\mathbf{y}):=\mathbb{E}_{\tau}(\mathcal{F}(\mathbf{y},\boldsymbol{\epsilon})|\mathbf{y}).
$$  

Remark A.1. In practice, the data may not lie on a manifold, but it is reasonable to assume that it lies on a tubular neighborhood of the manifold. Our notation accommodates this - if $\mathbf{z}$ is a point in a neighborhood of $\mathbb{X}$ , we may view it as a perturbation of a point $\mathbf{y}\in\mathbb{X}$ , so that the noisy value of the target function is $\mathcal{F}(\mathbf{y},\epsilon)$ , where $\epsilon$ encapsulate the noise in both the y variable and the value of the target function. $\mid$  

Our approximation process is simple: given by  

$$
\widehat{F}_{n,\alpha}(Y;\mathbf x):=\frac{n^{q(1-\alpha)}}{M}\sum_{j=1}^{M}\mathcal{F}(\mathbf{y}_{j},\epsilon_{j})\tilde{\phi}_{n,q}(n^{1-\alpha}|\mathbf x-\mathbf y_{j}|_{2,d}),\qquad\mathbf x\in\mathbb{R}^{d},
$$  

where $0<\alpha\le1$ .  

The theoretical underpinning of our method is described by Theorem A.1, and in particular, by Corollary A.1, describing the convergence properties of the estimator. In order to state these results, we need a smoothness class $W_{\gamma}(\mathbb{X})$ , representing the assumptions necessary to guarantee the rate of convergence of our estimator to the target function. It is beyond the scope of this paper to describe the details of this smoothness class; an interested reader will find them in [7]. We note that the definition of the estimator in (A.3) is universal, and does not require any assumptions. The assumptions are needed only to gurantee the right rates of convergence.  

Theorem A.1. Let $\gamma>0$ , $\tau$ be a probability distribution on $\mathbb{X}\times\Omega$ for some sample space $\varOmega$ such the marginal distribution of $\tau$ restricted to $\mathbb{X}$ is absolutely continuous with respect to $\mu^{*}$ with density $f_{0}\in W_{\gamma}(\mathbb{X})$ . We assume that  

$$
\operatorname*{sup}_{\mathbf{x}\in\mathbb{X},r>0}\frac{\mu^{*}(\mathbb{B}(\mathbf{x},r))}{r^{q}}\leq c.
$$  

Let $\mathcal{F}:\mathbb{X}\times\varOmega\rightarrow\mathbb{R}$ be a bounded function, $f$ defined by (A.2) be in $W_{\gamma}(\mathbb{X})$ , the probability density $f_{0}\in W_{\gamma}(\mathbb{X})$ . Let $M\geq1$ , $Y=\{(\mathbf{y}_{1},\epsilon_{1}),\cdots\,,(y_{M},\epsilon_{M})\}$ be a set of random samples chosen i.i.d. from $\tau$ . If  

$$
0<\alpha<\frac{4}{2+\gamma},\qquad\alpha\leq1,
$$  

then for every $n\geq1$ , $0<\delta<1$ and $M\geq n^{q(2-\alpha)+2\alpha\gamma}\sqrt{\log(n/\delta)}$ , we have with $\tau$ -probability $\geq1-\delta$ :  

$$
\left\|\widehat{F}_{n,\alpha}(Y;\circ)-f f_{0}\right\|_{\mathbb{X}}\leq c_{1}\frac{\sqrt{\|f_{0}\|_{\mathbb{X}}}\|\mathcal{F}\|_{\mathbb{X}\times\Omega}+\|f f_{0}\|_{W_{\gamma}(\mathbb{X})}}{n^{\alpha\gamma}}.
$$  

Corollary A.1. With the set-up as in Theorem $A.1$ , let $f_{0}\equiv1$ (i.e., the marginal distribution of y with respect to $\tau$ is $\mu^{*}$ ). Then we have with $\tau$ -probability $\geq1-\delta$ :  

$$
\left\|\widehat{F}_{n,\alpha}(Y;\circ)-f\right\|_{\mathbb{X}}\leq c_{1}\frac{\|\mathcal{F}\|_{\mathbb{X}\times\varOmega}+\|f\|_{W_{\gamma}(\mathbb{X})}}{n^{\alpha\gamma}}.
$$  