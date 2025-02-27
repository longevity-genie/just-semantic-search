# Review Article  

# Research progress on wearable devices for daily human health management  

Anan Li, Ping Shi\*  

School of Medical Instrument and Food Engineering, University of Shanghai for Science and Technology, Shanghai 200093, China. E-mail: garendon@163.com  

# ABSTRACT  

As the public’s demand for portable access to personal health information continues to expand, wearable devices are not only widely used in clinical practice, but also gradually applied to the daily health management of ordinary families due to their intelligence, miniaturization, and portability. This paper searches the literature of wearable devices through PubMed and CNKI databases, classifies them according to the different functions realized by wearable devices, and briefly describes the algorithms and specific analysis methods of their applications and made a prospect of its development trend in the field of human health.  

Keywords: wearable device; physiological signal; algorithm; sensor; health management  

# 1. Introduction  

Wearable devices, also known as wearable biosensors, can collect the original physiological parameters of the population, and then process them into health digital information that users can easily understand for health monitoring, such as heart rate, blood pressure, blood oxygen saturation, blood sugar and continuous monitoring of body temperature, etc. At the same time, the wearable device can also collect related indicators such as steps, activity category, posture, activity trajectory, sleep monitoring and energy consumption. Wear it on different parts of the human body according to the needs of users and the functions achieved by the device. The common wearing position and information transmission and storage process are shown in Figure 1.  

Compared with the early wearable devices, the wearable devices in recent years have been more lightweight, refined and fashionable in design. At the same time, as people’s demand for mobile health  increases,  higher  requirements  are  also placed on the performance of wearable devices. In order to better understand the application status of different wearable devices in health-related fields, this paper adopts the literature tracking method, and uses the PubMed and CNKI databases to search the literature in the past ten years using the keyword “wearable device”. Since this paper only studies the application of wearable devices in the field of human health, research in the fields of industry, education and military is excluded. The searched documents are classified by the application fields of wearable devices involved in the literature and the specific sensors and related algorithms used by the devices and summarises the relevant key technologies involved in the daily application of wearable devices, analyzes its possible problems, and summarizes the development trend of smart wearable devices in the field of human health.  

![](images/9fc3ec934fb277d226ef8d73d627e5ce8ddf40805d31fe60cc1bf83f08d5e229.jpg)  
Figure 1. Flow chart of common wearing positions and data transmission and storage of wearable devices.  

# 2. Application Status  

This paper uses the PubMed database to search the literature in the past ten years using the keyword “wearable device”, a total of 552 literature (excluding 908 review literature), through the statistical summary of the wearable devices used in the literature, according to the wearable device. The functions and application fields realized by wearable devices can be divided into the following categories.  

# 2.1. Entertainment and leisure  

Products such as smart glasses, wireless headsets and VR helmets are typical leisure and entertainment wearable devices, which occupy most of the wearable device market. In addition to the beautiful appearance, its core technology mainly lays in the battery life of the device, wireless communication technology and human-computer interaction effects. Generally, such wearable devices will combine multimedia applications such as cameras, videos, and music, and are mainly used for users to browse pictures, web pages, etc., and typical commercial products such as Google glass, Baidu Eye, Emotiv helmet, Apple watch, etc.  

# 2.2. Motion detection class  

Most  sports  detection  wearable  devices have  built-in  barometers,  three-axis  acceleration sensors and gyroscopes, which can detect the number of steps, distance, calorie consumption, activity type and posture during exercise. Due to the great instability of the signal during the movement, the sensitivity of the sensor and the accuracy of the algorithm  have  high  requirements.  Commercial wearable devices using different sensors are becoming more and more widely used, and the hottest one is in the field of motion analysis and activity monitoring of inertial measurement devices . Generally speaking, the filter, peak and valley detection and frequency domain adaptive threshold functions of sports wearable devices can make the device show strong stability for different users and environments. However, in most cases, few researchers have established a mathematical model of the relationship between sensor signals and activity detections.  

# Step count  

Step counting is the basic function of motion detection wearable devices, and the step counting function  is  mainly  realised  based  on  MEMS. Common step-counting detection algorithms mainly include  peak  detection  algorithm[2],  dynamic threshold detection algorithm, zero-rate correction algorithm, autocorrelation algorithm and combination of two or more algorithms. Other time- and frequency-based methods, such as fourier transform and wavelet transform, can utilize the walking cycle to achieve accurate step size detection. Studies have shown that commercial fitness wearable devices are less accurate in evaluating activity intensity than research-grade accelerometers[3]. Winfree et al.[4] evaluated Fitbit and found that Fitbit’s assessment of exercise intensity has a low accuracy rate. The team also used Actiongraph GT3X algorithm combined with Bayesian classifier to improve Fitbit Flex, reducing the error rate to $16.32\%$ . Tao et al.[5] made a review on a variety of pedometer APPs based on the Android system. The results show that the accuracy of step counting function of fitness APP is related to the actual walking speed and device placement. In general, a pedometer worn on the hip or foot is more accurate than a pedometer worn on the wrist or measured with a smart phone. Toth et al.[6] conducted a comparative analysis of 8 pedometers on the market (StepWatch, ActiPAL, Fitbit Zip, Yamax Digi-Walker SW-200, New Living Style NL-2000, Actiongraph GT9X, Fitbit Charge and Fitbit AG) using 14 different pedometer methods, and found that StepWatch has the highest pedometer accuracy. Various calibration methods can be used to improve the accuracy of the pedometer function of wearable devices, such as personalizing settings based on a single user and detecting the minimum walking duration before activating the pedometer function[7].  

# Energy expenditure  

The functions of wearable devices are gradually diversified, and calorie consumption is one of the focuses of consumers. Generally speaking, the measurement of energy expenditure (EE) includes direct measurement method and indirect measurement method, which can be represented by oxygen calorific value, respiration entropy, etc., and different formulas are used to calculate EE according to body mass, exercise time, speed, distance, etc. Among them, the double-labeled water method and the gas metabolism analysis method are  called  the  “gold  standard”  for  evaluating EE, but they are both expensive and inapplicable. Health-related smart wearable devices mostly use  

MotionX technology, which uses 3D accelerometers to identify motion patterns and convert them into identifiable energy consumption[8].  

Currently, there is no single technique that can accurately quantify EE under free-living conditions, but multiple methods can be combined to improve accuracy, such as heart rate, acceleration measurements, and step counts. Pande et al.[9] developed an initial linear regression model based on neural network and bagging regression tree, and the  correlation  between  the  EE  measured  by the barometer data and the actual EE measured by the gold standard calorimeter (COSMED K4b2) can reach $96\%$ . Shcherbina et al.[10] selected 60 volunteers to accept the evaluation of 7 devices in different states. The experiment showed that Apple Watch 3 had the lowest overall error rate, Samsung gear S2 has a high error rate, all devices have an error rate of more than $20\%$ . Most wrist-worn devices perform poorly on EE measurements during laboratory activities. The device is poor in measuring EE during laboratory activities. Some researchers  compared  three  commercial  sports watches  (Suinto  Ambit2,  Garmin  Forerunner920XT and Polar V800), and found that the calculation accuracy of the EE value of the device depends on the exercise intensity, and the error rate  of  the  three  devices  is  higher  under high-intensity exercise[11]. The accuracy of outdoor activity EE detection using wearable devices is still low, and more effective motion detection sensors and algorithms need to be developed[12].  

# Activity track and motion classification  

Human activity recognition systems can be roughly  divided  into  two  categories:  (1)  Systems based on computer vision; (2) systems based on acceleration sensors[13]. Various sensors can be used to improve the performance of the recognition system, such as RGB sensors, depth sensors (Kinect, etc.), and inertial sensors. Traditional motion recognition mainly follows the pattern shown in Figure $2^{[1]}$ .  

![](images/de6664934eef95282b5c48ed8b28bde22c8e7e87d1e5b41204117824afacd29c.jpg)  
Figure 2. Traditional motion recognition development system.  

Mooney et al.[14] evaluated two devices, Finis Swimsense $\textsuperscript{\textregistered}$ and Garmin SwimTM, The algorithms of  both  devices  can  accurately  assess  different strokes, but there are individual differences in accuracy (professional athletes have higher accuracy than amateurs). Kanoga’s team[15] used the EMG control system to identify motion through surface EMG, and found that compared with the traditional motion recognition algorithm, the armband-type surface EMG device has stronger performance for short-term use, but it is not suitable for long-term use. Commercial smart wearable devices mostly use GPS sensors to realize the positioning function, cooperate with three-axis sensors to realize the identification of different motion modes (climbing, walking, cycling, etc.), and use third-party applications to display dynamic motion trajectories in real time.  

# 2.3. Human health monitoring and medical applications  

Wearable  devices  can  not  only  provide short-term  physiological  data,  but  also  realize long-term and continuous human monitoring under different  conditions,  which  can  provide  a  certain  basis  for  clinical  decision-making[16,17].  Jovanov[18] found that 10 patients with chronic diseases who were intervened by wearable health monitoring devices had a significant decrease in average weight after 3 months, and their physical activity level and health status were significantly improved. Voss et al.[19] used Google glass to intervene in children with autism spectrum disorders, and taught children to recognize and express emotions through interventions such as pictures and audio from wearable devices. Clinical studies have found that wearable device-guided digital home therapy can improve current levels of care.  

# Sleep monitoring  

In general, polysomnography (PSG) is used as the gold standard for assessing sleep, but it is only suitable for clinical and laboratory research settings and requires professional operation. Actigraph is generally used in non-lab environment; this device can be worn all day, converts the collected physiological signals into digital signals and exports them through a three-axis accelerometer, etc. However, Actigraph has certain defects and cannot identify the sense of motion in a static state[20].  

Gautam et al.[21] classified the human body data  collected  by  the  built-in  accelerometer  of smart-phones based on the Kushida algorithm, statistical functions and hidden Markov models, and differentiated  between  sleep  and  wakefulness. Meltzer et al.[22] evaluated the sleep monitoring effect of Fitbit Ultra in 63 adolescents and children. The experiment proved that compared with PSG, Fitbit Ultra overestimated the total sleep time and sleep efficiency in normal mode, and the opposite in sensitive mode. Therefore, in clinical practice, this  device  cannot replace  traditional  PSG  and needs to be used with caution. Xie et al.[23] conducted  a  variety  of  functional  evaluations  on the best-selling and best-reviewed products on the market through a meta-analysis. In terms of sleep monitoring, these wearable devices achieved relatively high accuracy, with an average absolute percentage error of 0.11 and the difference between different devices is small. Previous studies have found that only using motion sensors to identify sleep states will produce large errors, and it is easy to classify resting and awake states as sleep states. Therefore, it is recommended to use multiple sensors to detect sleep. The introduction of new algorithms and parameters to evaluate sleep states may have better results[20].  

# Atrial fibrillation detection  

Atrial fibrillation is the most common arrhythmia disease, and is prone to complications such as arterial embolism, pulmonary embolism, cardiac insufficiency, and sudden cardiac death. Therefore, early prediction and timely treatment of atrial fibrillation are of great significance in reducing the incidence of stroke and other vascular embolic diseases[24].  

Common atrial fibrillation detection devices are mainly clinical ECG, implantable ECG equipment and portable wearable ECG measurement device.  Photoplethysmography  (PPG)  is  the  most common wearable technology used to detect cardiac function. Usually, the data from the PPG sensor is processed by the beat frequency detection algorithm, which  generally  includes  data  pre-processing, waveform extraction, peak and valley detection, and the classification of the interval between beats[25]. A new framework was proposed to distinguish atrial fibrillation from other types of heartbeats by combining an improved frequency slice wavelet transform with a convolutional neural network, confirming that it is possible to accurately identify atrial fibrillation from short-term signals[26]. There is also a portable ECG measurement device used in conjunction with ECG equipment. Fan et al.[27] used the “palm ECG” E-U08 device to remotely monitor the patient’s ECG outside the hospital and feed it back to the doctors in the hospital in real time. The detection rate is significantly higher than that of traditional 12-lead ECGs. William et al.[28] compared the Cartier mobile heart monitor with lead ECG in 52 patients, and confirmed that clinicians could improve the accuracy and efficiency of atrial fibrillation detection with the aid of equipment. The Kardia Band (KB), the first approved smart-watch accessory released by AliveCor, detects atrial fibrillation by recording single-lead ECG signals[29], and later released the Kardia Mobile 6L ECG device that can use six leads. The Study Watch can record, store and display ECG waveforms, but this watch can only be used for laboratory research and cannot provide user data access[30].  

Atrial fibrillation is a serious arrhythmia phenomenon, accompanied by various complications, and is the main cause of various heart diseases such as myocardial infarction. Atrial fibrillation is a serious arrhythmia phenomenon, accompanied by a variety of complications, is the main cause of myocardial infarction and other heart diseases, in order to realize automatic atrial fibrillation detection in small wearable devices, sophisticated sensors and algorithms are required, and the technical requirements are relatively high.  

# Fall detection  

Identifying falls and initiating early warning can effectively reduce related morbidity and mortality. Clinical testing is limited in time and space, and the equipment used is cumbersome. The fall detection function implemented by wearable devices is not prone to signal errors in practical applications, and is the most practical.  

General wearable devices are mostly based on three-axis devices such as accelerometers, gyroscopes and magnetometers, as well as multi-sensor fusion detection and video-based detection. Some studies have used machine learning methods to distinguish falls from normal states. Commonly used methods include $\mathbf{k}.$ -NN, least squares, support vector machines, Bayesian decision-making, dynamic time warping, and artificial neural networks[31]. There are also studies using statistical analysis to extract signal features to identify fall trends. Generally speaking, the risk of falling has individual differences, which  has  a  great relationship with age, body weight, etc., and the environment in which it is located also has a great influence on it.  

Analysis  methods  based  on  biomechanical models need to extract specific features, and the final model performance depends on the specific model structure and input data. Aicha et al.[32] compared the traditional biomechanical model and its proposed deep learning neural network model, and found that the latter’s fall risk prediction accuracy  

was significantly higher.  

In addition to the type and number of sensors, the placement of the sensors also has a great impact on  the  detection  effect.  The  Özdemir  group[33] summarized the number of sensors, subjects, sensor placement, sensor combination, classification algorithm and performance, the study found that simply reducing the number of sensors will reduce the detection accuracy, and the sensor using the k-NN algorithm is placed at the waist to achieve a sensitivity of $99.96\%$ .  

# Blood sugar test  

Diabetes is usually diagnosed and managed by continuous glucose monitoring (CGM) equipment. CGM can effectively control blood sugar and reduce insulin dosage in patients with type 2 diabetes mellitus[34]. In general, CGM provides input to a mathematical  model  that  predicts  fluctuations in blood glucose concentration over time. This algorithm relies on input from factors such as dietary intake, activity, and emotion that affect glucose metabolism, but is based on deep learning and support vector machines. The method can disregard these inputs and can predict the blood glucose change in patients with type 1 and type 2 diabetes for 60 minutes[35,36]. Bonn et al.[37] found that the intervention of the smart-phone APP combined with the GTX3X human exercise energy monitoring instrument in patients with type 2 diabetes can significantly improve the patient’s exercise volume and glucose and lipid metabolism indicators. Mhaskar et al.[38] used deep neural network to evaluate blood glucose in groups, and the results showed that compared with shallow network, the detection effect of deep neural network was better. There are few products for wearable devices to detect blood sugar, which is still an area to be studied and explored.  

# 3. Key technologies  

There are many kinds of wearable devices on the market, and the realization of different functions depends on different technical support, such as the sensors used in the device itself, external data receiving equipment, wireless communication technology and data storage platform.  

The  sensors  used in  wearable  devices  are mainly  divided  into  motion  transmission  sensors, biological sensors and environmental sensors, including gyroscopes, accelerometers, magnetometers, photoelectric sensors, barometric altimeters, and temperature sensors. Its human-computer interaction is different from ordinary smart devices. It is a direct and sufficient interaction method, mainly including voice interaction, tactile interaction, and consciousness interaction. At the same time, because wearable devices involve a wide range of fields, large amounts of data, and diverse application groups, it is necessary to use artificial intelligence to optimize the devices and platforms. Wireless communication technology is the link between users and devices, enabling data communication and information sharing between users, between users and devices, and between devices. Commonly used communication technologies now mainly include near field communication technology, Bluetooth and wireless network technology. Users can transmit data to the cloud platform for subsequent viewing, use and sharing through wireless communication technology with low energy consumption. Compared  with  the  traditional  human-computer interaction mode, the application of virtual reality and augmented reality in the wearable field pays more attention to the actual feeling of human senses. The way to obtain information is no longer limited by time and space, and virtual screens may become a visual supplement for human-computer interaction.  

# 4. Conclusions  

Most of the traditional wearable devices are devices based on research institutions or medical places guided by special personnel, providing real-time visual physiological data for specific users. As people pay more attention to health, the concept of smart medical care is more deeply rooted in the hearts of the people. Due to the limited medical resources, wearable devices also mean the transformation to the field of individual medical applications. It must develop in the direction of more informatization, digitization and intelligence[39]. Due to the development of technologies such as sensors, external data receiving devices, wireless communication technology and data storage platforms used in the device itself, in addition to the ordinary motion detection function, most smart wearable devices currently have certain human health management functions and the reliability of the detection data is  

# high.  

However, the popularization of smart wearable devices still faces a series of problems and challenges (Table 1). The development of Internet technology has made people not only have high requirements for device signal reliability, long-term stability and comfort, but also require data. There is also more attention to privacy protection, and it is necessary to continuously improve the algorithms for processing signals and analyzing data[42,43].  

Table 1. The application status of wearable devices   

![](images/39b166a46be0d5356e31af3113de8072e95566b943cca8d8bb5a270ccd371de9.jpg)  

Most wearable devices are not very independent, and need corresponding terminal APP support. At the same time, the portable characteristics also require the miniaturization and integration of the sensors of the wearable device, and also require the device to have a certain battery life. Due to different application fields, there is still a lack of unified standards for general smart wearable devices. Although there are many types of wearable devices on the market with complex functions, however, there is still a lot of controversy about the application of special groups (the elderly, children and pregnant women, etc.).  

Although there are many challenges, the development  of  portable  smart  wearable  devices has become a major trend. With the development of technology, the hardware technology of the device (processor, battery technology, etc.), software system (user-centric more accurate algorithms, etc.), cloud  services  (personalized  services,  etc.)  will achieve a certain degree of performance improvement, and the user experience will also be significantly enhanced. At the same time, with the development of 5G technology[44], the application of communication Internet technology will be more in-depth, providing technical supplements for the scarce medical resources in the post-epidemic era.  

This trend will promote the cross-integration of expertise in more fields, promote the coordinated development of various industries, and will also create a healthier and safer application environment for smart wearable devices.  

# Conflict of interest  

The authors declare no conflict of interest.  

# References  

1. Jalloul N, Poree F, Viardot G, et al. Activity recognition using complex network analysis. IEEE Journal of Biomedical and Health Informatics 2018; 22(4): 989–1000.   
2. Ahola TM. Pedometer for running activity using accelerometer  sensors  on  the  wrist.  Medical Equipment Insights 2010; (3): 1–8.   
3. Gaglani S, Moore J, Haynes MR, et al. Using commercial activity monitors to measure gait in patients with suspected iNPH: Implications for ambulatory monitoring. Cureus 2015; 7(11): e382.   
4. Winfree KN, Dominick G. Modeling clinically validated physical activity assessments using commodity hardware. 2017 IEEE EMBS International Conference on Biomedical and Health Informatics; 2012 Jan 5-7 Jan; Orlando: IEEE Press; p. 157–160.   
5. Tao R, Zhang C. Research on the validity of the step counting function of fitness APP in the era of artificial  intelligence.  Abstracts  of  the $11^{\mathrm{th}}$ National Sports Science Conference; 2019 Nov 1–3; Nanjing: Chinese Society of Sports Science. p. 5468–5470.   
6. Toth LP, Park S, Springer CM, et al. Video-recorded validation  of  wearable  step  counters  under free-living conditions. Medicine and Science in Sports and Exercise 2018; 50(6): 1315–1322.   
7. Bassett DR, Toth LP, LaMunion SR, et al. Step counting: A review of measurement considerations and health-related applications. Sports Medicine 2017; 47(7): 1303–1315.   
8. EI Arawy F, Nounou MI. Are currently available wearable devices for activity tracking and heart rate monitoring accurate, precise, and medically beneficial? Healthcare Informatics Research 2015; 21(4): 315–320.   
9. Pande A, Zhu J, Das AK, et al. Using smart-phone sensors for improving energy expenditure estimation. IEEE Journal of Translational Engineering in Health and Medicine 2015; 3: 270021.   
10.  Shcherbina A, Mattsson CM, Waggott D, et al. Accuracy in wristworn, sensor-based measurements of heart rate and energy Expenditure in a diverse cohort. Journal of Personalized Medicine 2017; 7(2): 3.   
11.  Roos L, Taube W, Beeler N, et al. Validity of sports watches when estimating energy expenditure during running. BMC Sports Science, Medicine and Rehabilitation 2017; 9(1): 22.   
12.  Ryu JS, Park JJ. Validation of wearable devices to measure energy consumption. The Asian Journal of Kinesiology 2020; 22(1): 33–37.   
13.  Guo YN, Tao DP, Liu WF, et al. Multiview cauchy estimator feature embedding for depth and inertial sensor-based  human  action  recognition.  IEEE Transactions on Systems, Man, and Cybernetics: Systems 2017; 47(4): 617–627.   
14.  Mooney R, Qunilan LR, Corley G, et al. Evaluation of the Finis Swimsense (R) and the Garmin Swim (TM) activity monitors for swimming performance and stroke kinematics analysis. PloSOne 2017; 12(2): e0170902.   
15.  Kanoga S, Kanemura A, Asoh H. Are armband sEMG devices dense enough for long-term use? Sensor placement shifts cause significant reduction in recognition accuracy.  Biomedical Signal Processing and Control 2020; 60: 101981.   
16.  Gualtieri L, Rosenbluth S, Phillips J. Can a free wearable activity tracker change behavior? The impact of trackers on adults in a physician-led wellness group. JMIR Research Protocols 2016; 5(4): e237.   
17.  De Brouwer M, Ongenae F, Bonte P, et al. Towards a cascading reasoning framework to support responsive  ambient-intelligent  healthcare  interventions. Sensors 2018; 18(10): 3514.   
18.  Jovanov E. Wearables meet IoT: Synergistic personal area networks (SPANs). Sensors 2019; 19(19):   
4295.   
19.  Voss C, Schwartz J, Daniels J, et al. Effect of wearable digital intervention for improving socialization in children with autism spectrum disorder a randomized clinical trial. JAMA Pediatrics 2019;   
173(5): 446–454.   
20.  De Zambotti M, Cellini N, Goldstone A, et al. Wearable sleep technology in clinical and research settings. Medicine and Science in Sports and Exercise 2019; 51(7): 1538–1557.   
21.  Gautam  A,  Naik  VS,  Gupta  A,  et  al.  An smartphone-based algorithm to measure and model quantity of sleep. The $7^{\mathrm{th}}$ International Conference on Communication Systems and Networks; 2017 Nov 11–13; Bangalore: IEEE Press; p. 1–6.   
22.  Meltzer LJ, Hiruma LS, Avis K, et al. Comparison of a commercial accelerometer with polysomnography and actigraphy in children and adolescents. Sleep   
2015; 38(8): 1323–1330.   
23.  Xie J, Wen D, Liang L, et al. Evaluating the validity of current mainstream wearable devices in fitness tracking under various physical activities: Comparative study. JMIR mHealth and uHealth 2018; 6(4): e94.   
24.  Wang Y, Zhang S, Xu W. Portable and wearable ECG measurement devices in atrial fibrillation application in screening.  Chinese Medical Journal   
2020; 55(06): 592–593.   
25.  Aboy M, McNames J, Thong T, et al. An automatic beat detection algorithm for pressure signals. IEEE Transactions on Biomedical Engineering 2005;   
52(10): 1662–1670.   
26.  Xu X, Wei S, Ma C, et al. Atrial fibrillation beat identification using the combination of modified frequency slice wavelet transform and convolutional neural networks. Journal of Healthcare Engineering   
2018; 2018: 2102918.   
27.  Fan P, Chen C, Peng Y, et al. Clinical application of E-U08 snap ECG recorder in remote monitoring of arrhythmia and myocardial ischemia. Journal of Practical Electrocardiology 2017; 26(1): 10–15.   
28.  William AD, Kanbour M, Callahan T, et al. Assessing the accuracy of an automated atrial fibrillation detection algorithm using smartphone technology: The iREAD study. Heart Rhythm 2018; 15(10):   
1561–1565.   
29.  Bumgarner JM, Lambert CT, Hussein AA, et al. Smartwatch algorithm for automated detection of atrial fibrillation. Journal of the American College of Cardiology 2018; 71(21): 2381–2388.   
30.  January CT, Wann LS, Calkins H, et al. 2019 AHA/ACC/HRS  focused  update  of  the  2014 AHA/ACC/HRS guideline for the management of patients  with  atrial  fibrillation:  A  report of  the American College of Cardiology/American Heart Association task force on clinical practice guidelines and the heart rhythm society. Circulation 2019;   
140(2): e125–e151.   
31.  Zhao X, Hu A, He W. Fall detection based on convolutional neural network and XGBoost. Las Optoelect Prog 2020; 57(16): 16102.   
32.  Aicha AN, Englebienne G, Van Schooten KS, et al. Deep learning to predict falls in older adults based on daily-life trunk  accelerometry.  Sensors 2018;   
18(5): 1654.   
33.  Özdemir AT. An analysis on sensor locations of the human body for wearable fall detection devices: Principles and practice. Sensors 2016; 16(8): 1161.   
34.  Zhen X, Tian J, Shi Z, et al. Analysis on the application value of wearable devices in blood glucose management of diabetic patients. China Modern Doctor 2020; 58(10): 100–103.   
35.  Hamdi T, Ali JB, Costanzo VD, et al. Accurate prediction of continuous blood glucose based on support vector regression and differential evolution algorithm. Biocybernetics  and Biomedical Engineering 2018; 38(2): 362–372.   
36.  Mhaskar HN, Pereverzyev SV, Van Der Walt MD. A deep learning approach to diabetic blood glucose prediction. Frontiers in Applied Mathematics and Statistics 2017; 3: 14.   
37.  Bonn  SE,  Alexandrou  C,  Steiner  KH,  et  al. App-technology to increase physical activity among patients with diabetes type 2-The DiaCert-study, a randomized controlled trial. BMC Public Health   
2018; 18: 119.   
38.  Jia N, Li Y. Construction of personalized health monitoring platform based on intelligent wearable devices. Computer Science 2019; 46(z1): 566–570.   
39.  Schutz Y, Weinsier RL, Hunter GR. Assessment of free-living physical activity in humans: An overview of currently available and proposed new measures. Obesity Research 2001; 9(6): 368–379.   
40.  Perez  MV,  Mahaffey  KW,  Hedlin  H,  et  al. Large-scale assessment of a smartwatch to identify atrial fibrillation. The New England Journal of Medicine 2019; 381(20): 1909–1917.   
41.  Gilani M, Eklund JM, Makrehchi M. Automated detection of atrial fibrillation episode using novel heart rate variability feature. The $38^{\mathrm{th}}$ Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC); 2016 Aug 16–20; Orlando: IEEE Press; p. 3461–3464.   
42.  Hu K, Chen X, Zhang S, et al. Research and application of wearable devices in rehabilitation medicine in developed countries. China Digital Medicine 2018; 13(8): 56–59, 15.   
43. Kou J. Deepening the application of smart wearable devices in the medical and health field. People’s Posts and Telecommunications, $2020\;\mathrm{Jun}\;12$ .   
44. Wu X, Wu Y, Huang X, et al. Study on the optimization of health management model for the elderly in urban communities under the background of smart medical treatment. China Medical Herald 2020; 17(33): 194–196.  