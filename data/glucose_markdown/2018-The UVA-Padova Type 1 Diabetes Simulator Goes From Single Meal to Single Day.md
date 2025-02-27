# The UVA/Padova Type 1 Diabetes Simulator Goes From Single Meal to Single Day  

Roberto Visentin, PhD1\*  , Enrique Campos-Náñez, PhD2\* Michele Schiavon, PhD1, Dayu Lv, PhD2, Martina Vettoretti, PhD1, Marc Breton, PhD2, Boris P. Kovatchev, PhD2, Chiara Dalla Man, PhD1, and Claudio Cobelli, PhD1  

# Abstract  

Background: A new version of the UVA/Padova Type 1 Diabetes (T1D) Simulator is presented which provides a more realistic testing scenario. The upgrades to the previous simulator, which was accepted by the Food and Drug Administration in 2013, are described.  

Method: Intraday variability of insulin sensitivity (SI) has been modeled, based on clinical T1D data, accounting for both intra- and intersubject variability of daily $\mathsf{S}_{\mathrm{l}}$ . Thus, time-varying distributions of both subject’s basal insulin infusion and insulin-to-carbohydrate ratio were calculated and made available to the user. A model of “dawn” phenomenon based on clinical T1D data has been also included. Moreover, the model of subcutaneous insulin delivery has been updated with a recently developed model of commercially available fast-acting insulin analogs. Models of both intradermal and inhaled insulin pharmacokinetics have been included. Finally, new models of error affecting continuous glucose monitoring and selfmonitoring of blood glucose devices have been added.  

Results: One hundred in silico adults, adolescent, and children have been generated according to the above modifications. The new simulator reproduces the intraday glucose variability observed in clinical data, also describing the nocturnal glucose increase, and the simulated insulin profiles reflect real life data.  

Conclusions: The new modifications introduced in the T1D simulator allow to extend its domain of validity from “singlemeal” to “single-day” scenarios, thus enabling a more realistic framework for in silico testing of advanced diabetes technologies including glucose sensors, new insulin molecules and artificial pancreas.  

# Keywords  

computer simulation, diabetes control, modeling  

In the last decades, computer simulation has allowed important steps forward in type 1 diabetes (T1D) research, enabling the possibility to perform several in silico tests, with relevant time- and cost- savings. Several simulation tools have been developed (for a review, see Cobelli et al,1 Kovatchev et al,2 Wilinska et al,3 Kanderian et al4), each one based on a comprehensive mathematical model and equipped with an in silico population. In particular, in 2008 the US Food and Drug Administration (FDA) accepted the T1D simulator developed by Universities of Virginia (UVA) and Padova as a substitute for preclinical trials for certain insulin treatments, including closed-loop algorithms for artificial pancreas (AP). This dramatically accelerated the AP research in humans.  

Since its first release (S2008), the simulator has been equipped  with  an  in  silico  population  of  300  subjects spanning the variability observed in the real T1D population. This represents the key strength of this “new generation” simulator: in fact in silico subjects are generated from a robust joint distribution of model parameters, obtained by identifying a rather complex model,5 from a unique multiple-tracer dataset in healthy subjects, which was adapted to describe the T1D variability available in the literature.  

![](images/6ebaf7a5cce471f31fa35852c2f286bfc832637652a27a29485d13bf2f1e6bcf.jpg)  
Figure 1.  Plasma glucose profile in response to three identical meals, simulated with a time-invariant (dashed red line) or a time-varying model (continuous green line), respectively. A time-varying model provides more realistic behavior, for which glucose response to the same meal perturbation can be different during the day.  

A first update of the simulator (S2013) was released and accepted by FDA in 2013, which incorporated a nonlinear glucose response to hypoglycemia and a model of counterregulation.7 S2013 has been assessed against clinical T1D data,8 demonstrating that the in silico population was indeed representative of T1D clinical data; this proved the robustness of the simulator in the single-meal scenario domain of validity.  

Both S2008 and S2013 have been extensively used by more than 30 sites in academia and companies involved in T1D research, more than 70 articles were published in peerreviewed journals, and, despite the “single-meal” domain of validity, the simulator was employed for testing AP algorithms which have been subsequently tested in weekly and monthly outpatient human trials.9-16  

However, the latest technological advances in T1D research call for more realistic testing scenarios than the single-meal framework, for example, adaptive AP prototypes17-22 and slow-acting insulins have to be evaluated on single/multiple day scenarios. Both S2008 and S2013 do not cope with this requirement, since, for example, they provide the same meal response throughout the day. In contrast, the simulator should be able to describe a series of factors that could affect glucose metabolism during the day, such as daily variations in subject parameters, different composition of ingested meals, physical activity, and illness. Therefore, the resulting glucose response to a certain perturbation, for example, a meal, should be strictly dependent from the time at which the perturbation occurs (Figure 1).  

The ability to well reproduce diurnal variability is thus of interest, and other research groups have worked on this topic.4,23-25 We recently demonstrated the ability of the T1D simulator model to describe glucose dynamics in a large dataset of T1D subjects studied for about 24 hours, by describing insulin sensitivity (SI) and meal intestinal absorption rate with time-varying parameters.26 Such a choice was based on the notion that circadian variability of $\mathrm{S}_{\mathrm{I}}$ is well known to occur in T1D subjects, as recently demonstrated by Hinshaw and colleagues,27 and on the assumption that meal absorption rate is depending from its composition.  

In this regard, we recently developed a model of intraday $\mathrm{S}_{\mathrm{I}}$ variability,28 and incorporated it into the simulator, thus making it time-varying and extending its domain of validity from “single-meal” to “single-day” (24-hour scenario).  

In this article, we present the new version of the UVA/ Padova T1D simulator (S2017), which incorporates the key feature described above, together with other novelties, such as models describing the latest developments in glucose measurement devices, insulin analogues, and routes of administration (Figure 2).  

# Methods  

# Time-Varying Model of T1D Subject  

The most important feature of S2017 concerns the model of T1D subject (shown in Figure 2; see the appendix for the complete set of model equations), which incorporates some time-varying model parameters that allow the description of intrasubject diurnal glucose variability.  

Intraday Variability of Insulin Sensitivity.  A study in T1D proved the existence of diurnal patterns in $\mathrm{~\dot{S}_{L}}^{27}$ Briefly, 20 T1D subjects underwent a mixed meal with triple-tracer approach at breakfast (B), lunch (L), and dinner (D), with all meals having the same amount and composition. This allowed a reliable estimation of $\mathrm{S}_{\mathrm{I}}$ at B, L and D, by eliminating possible confounding effects due to different meal composition or size, thus relating the observed intraday glucose variability to the sole $\mathrm{S}_{\mathrm{I}}$ variations. Results showed that $\mathrm{S}_{\mathrm{I}}$ was lower, on average, at B than $\mathrm{L}$ and D, but the intersubject variability was large. This knowledge has been incorporated into the model:28 specifically, each in silico subject is associated with one among seven possible variability classes, each one representing a specific $\mathrm{S_{I}}$ daily pattern, based on the probabilities reported in Visentin et  al;28 then, the time-varying $\mathrm{S}_{\mathrm{I}}$ is implemented by varying in time parameters $k_{p3}$ and $V_{m x}$ (see equations (A5) and (A10), respectively), that is, the insulin action on glucose production by the liver and on glucose utilization by tissues, respectively.  

Dawn Phenomenon.  Nocturnal glucose variability is modeled to describe the so-called “dawn” phenomenon, that is, the rise in blood glucose concentration that occurs in T1D subjects during early morning, due to both an increased endogenous glucose production $(E G P)$ and an increased insulin requirements.29,30 Recently, Mallad et al29 quantified an almost $30~\mathrm{mg/dL}$ increase in glucose concentration from $3{\mathrel{:}}00{\mathrm{~AM}}$ to $7{\because}00\,\mathrm{AM}$ and observed an increase of $E G P$ of about $1.5~\mathrm{mg/kg/min}$ in the same interval. Thus, we modeled the $E G P$ variation as a linear increase of $E G P$ at zero glucose and insulin (represented in the model by parameter $k_{p I}$ of equation (A5)) from 3:00 am to 7:00 am. Similarly, the increased insulin requirement is described by a decrease in insulin-dependent glucose utilization (through a parameter $k_{i r}$ of equation (A10)). Intersubject variability of both time interval and magnitude of $k_{p I}$ increase and $k_{i r}$ decrease are obtained as random modulation of the averages reported in Mallad et al.29  

![](images/b5f6656ba6ad650f142f15ecb7b9ea24718db870d48269261eeac067ccff3709.jpg)  
Figure 2.  Scheme of T1D S2017 highlighting the introduced upgrades. Specifically, the model incorporates time-varying parameters describing intraday $\mathsf{S}_{\mathrm{l}}$ variability $(k_{p3},V_{m x})$ and dawn phenomenon $(k_{p\,I},\,k_{i r})$ ; insulin delivery routes account for administration of subcutaneous fast-acting insulin, intradermal and inhaled insulins; glucose monitoring devices include models of both CGM and SMBG.  

Updated Glucagon Model.  With respect to S2013, S2017 incorporates two updates in the model of glucagon secretion, kinetics and action: first, glucagon clearance, originally expressed as absolute rate $\mathrm{(min^{-1})}$ , is now fractional, that is, per unit of distribution volume $\mathrm{(mL/kg/min)}$ , in order to provide an easy comparison with the literature;31 second, no static secretion is assumed when blood glucose is over its basal value (see equation (A25)), in order to avoid nonphysiological oscillations in simulated glucagon concentration.  

# Insulin Delivery Models  

Subcutaneous administration of fast-acting insulin analogs is part of conventional insulin therapy in T1D, and is thus a fundamental component of the T1D simulator. However, in order to accommodate alternative insulin delivery routes, S2017 also incorporates models of intradermal and inhaled insulins (Figure 2).  

Subcutaneous  Fast-acting  Insulin.  S2017  incorporates  an improved model of subcutaneous insulin kinetics, which is able to reproduce the commercially fast-acting insulin analogs.32 This model has been identified on a large dataset of 116 T1D subjects and introduces, with respect to that included in S2008 and S2013, a subject-specific delay in insulin absorption (τ), accounting for both removal of auxiliary substances for insulin storage in the vial and diffusion processes occurring in the subcutis (see equations A16, A17).  

Intradermal Insulin.  The model of intradermal insulin route is based on Lv et al.33 This model (equations A18, A19) has been identified on 10 healthy subjects, and assumes that insulin delivered at the intradermal site is then transported to the bloodstream via two independent routes, that is, a diffusion-like process and a combination of a diffusion-like processes followed by an additional compartment before entering the blood.  

Inhaled Insulin.  The pulmonary route of administration is described using the model of inhaled insulin pharmacokinetic described in Visentin et  al.34 This model (equations A20, A21) has been identified on 12 T1D subjects, and assumes that a certain fraction of inhaled insulin reaches the alveoli, and then it is rapidly absorbed from the lungs.  

# Glucose Monitoring Models  

S2017 implements the most up-to-date technology for glucose sensing, that is, models describing both continuous glucose  monitoring  (CGM)  and  self-monitoring  of  blood glucose (SMBG) devices (Figure 2). These models, already presented in Vettoretti et  al,35,36 Facchinetti et  al,37 and Campos-Náñez et al,38 are briefly described below.  

CGM Model.  CGM measurements are modeled to reproduce the performance of the Dexcom $\textsuperscript{\textregistered}$ G5 Mobile device (Dexcom, Inc, San Diego, CA). In particular, CGM model derives from that developed by Facchinetti et al:37 ideal interstitial glucose (IG) signal is derived from simulated blood glucose (BG) profile; then, sensor calibration error is modeled with a linear polynomial; finally, an additive measurement noise generated by a second-order autoregressive model is added. Parameters of the calibration error model and the sensor noise model are randomly sampled, for each virtual sensor, from a day-specific multivariate probability density function derived by fitting the model to data collected in 76 subjects wearing the Dexcom G5 Mobile device for 7 days.  

SMBG Model.  SMBG model has been developed to reproduce the performance of the Bayer $\textsuperscript{\textregistered}$ Contour Next USB device (Bayer HealthCare LLC, Diabetes Care, Whippany, NJ). Specifically, the SMBG model, proposed by Vettoretti et al,35 consists of a combination of a skew-normal density function and two exponential functions, and is used to describe the probability density functions of both absolute and relative errors, respectively in the low $\mathrm{{(BG\leq115~mg/dl)}}$ and the high glucose range $\mathrm{(BG>}115~\mathrm{mg/dl)}$ ). The model was developed and validated using data collected in 51 subjects (44 T1D and 7 with type 2 diabetes). The simulator also incorporates models of 52 commercially available models previously described in Campos-Náñez et al.38 The models provide performance similar to published data on these meters in terms of percentage of measurements within ranges in high and low glucose ranges $(>\!100\:\mathrm{mg/dL}$ , ${<}100\;\mathrm{mg/dL})$ .  

# In Silico Population  

S2017 is equipped with an in silico population of 300 virtual subjects: 100 adults, 100 adolescents, 100 children. As in Dalla Man et al,7 each in silico subject is represented by a vector of model parameters that has been randomly extracted from a joint parameter distribution (i.e., covariance matrix and average vector of model parameters).  

The novelty here is that each in silico subject is equipped with a daily pattern of basal insulin rate that minimizes glucose oscillations caused by $\mathrm{S}_{\mathrm{I}}$ variability and dawn phenomenon;  in  particular,  the  optimal  insulin  requirement  is calculated from model equations, assuming steady state conditions at early morning, breakfast, lunch, and dinner, respectively. Similarly, a diurnal pattern of carbohydrate-to-insulin ratio (CR) is provided for each subject, containing the optimal CR calculated at breakfast, lunch, and dinner, by adopting the same strategy described in Dalla Man et al.7 Total daily insulin (TDI) is determined from basal infusion rate and optimal CR, assuming an average diet of $180~\mathrm{g}$ of CHO for adults and adolescents, and $135\;\mathrm{g}$ for children; correction factor (CF) is then calculated using the so-called 1700 rule.39  

Following Dalla Man et al, some additional criteria have been considered for subject generation, that is, Mahalanobis distance lower than the $95\%$ percentile and steady state glucose in absence of insulin infusion $>300\;\mathrm{mg/dL}$ . On the other hand, constraints on CR values have been enlarged in order to cope with the time-varying $\mathrm{S_{I}}$ , which can exceed the $100\%$ of the originally generated value; in particular, here we set $\mathrm{CR}\leq$ $40\:\mathrm{g/U}$ for adult and adolescents, $\mathrm{CR}\leq50~\mathrm{g/U}$ for children.  

# Simulation Environment  

S2017 is implemented in MATLAB $\textsuperscript{\textregistered}$ R2016. The software interface is similar to that of the previous S2013. In particular, the user is allowed to configure the simulation in terms of: (1) experiment attributes, such as simulation length, meal times, and CHO amounts; (2) population age and number of subjects undergoing the simulation; (3) hardware for glucose reading, such as a specific CGM sensor or SMBG, and insulin administration route, like subcutaneous, intradermal or inhaled; (4) results to be displayed, for example, outcome metrics of glucose control and variability, and graphs.  

# Results  

One hundred in silico adults, adolescent, and children have been generated using the criteria described above. Average (standard deviation, SD), minimum, and maximum values of the key metabolic parameters in the three cohorts are reported in Table 1. In silico CR values at breakfast are lower than those at lunch and dinner within each cohort, in agreement with Hinshaw et al.27  

The distributions of CR at breakfast, lunch, and dinner, together with that of CF in the three populations are also shown in Figure 3. They describe accurately the distribution observed in real patients. In particular, CFs in children are higher than in adolescents and adults, reflecting both the higher insulin sensitivity and the lower body weight.  

Figure 4 shows 24-hour profiles of plasma glucose and insulin concentrations of 100 adults, adolescents, and children receiving $60\,\mathrm{g}$ of CHO at $7{\because}00\,\mathrm{AM}$ , 1:00 pm, 7:00 pm, and the optimal insulin bolus (according to patient’s CR and CF).  

Table 1.  Key Demographic and Metabolic Parameters of the In Silico Subjects Available in the Simulation Environment.   

![](images/c24f9b4101fd4e0fcc960ca609ad697a1d767c7ce441a6b053a9a19122863934.jpg)  

![](images/72a4b1219bbc07316b24e90a624352429299e6e565fb8374f90857b07ee94df0.jpg)  
Figure 3.  Simulated distribution of carbohydrate-to-insulin ratio at breakfast $(\mathsf{C R}_{\mathsf{B}})$ , lunch $(\mathsf{C R}_{\mathsf{L}})$ , dinner $(\mathsf{C R}_{\mathsf{D}})$ and correction factor (CF), respectively, from left to right panels, in adult (upper), adolescent (middle), and children (lower panels) populations.  

The observed intersubject glucose variability is larger in children and adolescents compared to adults, in agreement with the knowledge that glucose control is more challenging in these groups of subjects. Moreover, in all the cohorts the postprandial glucose and insulin excursions at breakfast (B) are higher, on average, with respect to those at lunch (L) and dinner (D). This reflects the fact that several subjects are insulin resistant in that portion of the day: mean $\pm\,\mathrm{SD}$ of overbasal glucose peak values are $68\pm30$ (B), $54\pm27$ (L), and $53\pm25$ (D) $\mathrm{mg/dL}$ in adults, $98\pm36$ (B), $79\pm38$ (L), and $73\pm34\,(\mathrm{D})\,\mathrm{mg/dL}$ in adolescents, $134\pm45$ (B), $117\pm42$ (L), and $114\pm42\,(\mathrm{D})\,\mathrm{mg/dL}$ in children; mean $\pm$ SD of overbasal insulin peak values are $185\pm108$ (B), $138\pm92$ (L), and $133\pm80\,(\mathrm{D})\,\mathrm{pmol/L}$ in adults, $333\pm185$ (B), $244\pm137$ (L), and $229\pm130$ (D) pmol/L in adolescents, $422\pm$ 236 (B), $325\pm179$ (L), and $315\pm180\,(\mathrm{D})\,\mathrm{pmol/L}$ in children.  

# Discussion  

Since its first release in 2008, the UVA/Padova T1D simulator has been widely used in AP research, contributing to speed up the transition of AP testing from short to long-lasting periods, such as multiple days/weeks. However, both 2008 and 2013 simulator versions cannot fully support the contemporary long-term AP trials due to their single-meal domain of validity: in fact these simulator versions are based on time-invariant parameters. On the contrary, recent works demonstrated that simulation models need for a certain number of time-varying parameters in order to properly describe glucose metabolism during the day.26,40 Therefore, we developed a model of $\mathrm{S}_{\mathrm{I}}$ intraday variability and a description of dawn phenomenon, thanks to available data on diurnal variability of glucose metabolism. These models have been incorporated into the new version of the UVA/Padova T1D simulator (S2017), presented in this article. A new in silico population  has  been  generated,  where  each  subject  is equipped with a time-varying diabetes therapy, consisting of a time-varying basal insulin rate and multiple CRs, which better complies with conventional therapy of real T1D subjects. In addition, S2017 incorporates new models implementing the latest technological advances in insulin delivery and glucose sensing.  

![](images/4abfa52cbfe57459769ccefda01f14d9ba240f85ae3a6b5ed6d1f702517f5a5a.jpg)  
Figure 4.  Simulated plasma glucose (upper) and insulin (lower panels) in the 100 in silico adults (left), adolescents (middle), and children (right panels) undergoing a 24-hour scenario with three identical meals ( $_{60\,\mathfrak{g}}$ of CHO) at $7{\tt{:}00}\textsf{A M}$ , $1{\mathopen{:}}00\ p m$ , $7{\cdot}00\ p m$ , respectively, and receiving optimal subcutaneous insulin basal and bolus.  

Although the described upgrades increase the reliability of the simulator, further improvements are already envisioned and under investigation. In Visentin et  al,26 a preliminary description of glucose dynamics depending on meal composition was provided; however, a better insight on this topic will be made based on a recent study on complex carbohydrates.41 We also foresee the possibility of including multiple daily injection therapy in the next version of the simulator, by incorporating pharmacokinetic models of long-acting basal insulin analogues, such as that describing the absorption of glargine insulins.42 Finally, new insights based on data available from recent monthly clinical trials, for example,11-13,21 will allow an appropriate description of long-term $\mathrm{S}_{\mathrm{I}}$ variability, thus potentially further extending the simulator domain of validity from single- to multiple-day scenarios.  

# Conclusion  

In this article we presented the new version of the UVA/ Padova T1D Simulator. It incorporates a number of novelties that are based on solid modeling methodologies and original data. Among them, the most noteworthy upgrade concerns the model of diurnal glucose variability, mainly due to the intraday $\mathrm{S}_{\mathrm{I}}$ variability, which allows extending the domain of validity of the simulator from single-meal to single-day multiple-meal scenarios. The 2017 updated simulator will be submitted to FDA for acceptance. The new simulator has the potential of being a suitable tool for in silico testing of the latest cutting-edge technologies in type 1 diabetes therapy.  

# Appendix  

# Model Equations  

Glucose Subsystem  

$$
\left\{\begin{array}{l l}{\dot{G}_{p}\left(t\right)=E G P\left(t\right)+R a_{m e a l}\left(t\right)-U_{i i}\left(t\right)-E\left(t\right)}\\ {\quad\quad\quad-k_{1}\cdot G_{p}\left(t\right)+k_{2}\cdot G_{t}\left(t\right)}\\ {G_{p}\left(0\right)=G_{p b}}\\ {\dot{G}_{t}\left(t\right)=-U_{i d}\left(t\right)+k_{1}\cdot G_{p}\left(t\right)-k_{2}\cdot G_{t}\left(t\right)}\\ {G_{t}\left(0\right)=G_{t b}}\\ {G(t)=G_{p}\left(t\right)/V_{G}}\\ {G(0)=G_{b}}\end{array}\right.
$$  

Insulin Subsystem  

$$
\begin{array}{r l}&{\left[\dot{I}_{p}\left(t\right)\!=\!-\!\left(m_{2}+m_{4}\right)\!\cdot\!I_{p}\left(t\right)\!+m_{1}\cdot I_{l}\left(t\right)\!+R a_{I}\left(t\right)\right.}\\ &{\left.\!\left|I_{p}\left(0\right)\!=\!I_{p b}}\\ &{\dot{I}_{l}\left(t\right)\!=\!-\!\left(m_{1}+m_{3}\right)\!\cdot\!I_{l}\left(t\right)\!+m_{2}\cdot I_{p}\left(t\right)}\\ &{\left.\!\!I_{l}\left(0\right)\!=\!I_{l b}}\\ &{\left.\!\!I\left(t\right)\!=\!I_{p}\left(t\right)\!/V_{I}}\\ &{\left.\!\!I\left(0\right)\!=\!I_{b}}\end{array}
$$  

# Glucose Rate of Appearance  

$$
\begin{array}{r l}&{\left(Q_{u v}\left(t\right)-Q_{u v}\left(t\right)+Q_{u v2}\left(t\right)\right.}\\ &{\left.\left(Q_{u v}\left(0\right)=0\right.\right.}\\ &{\left.\left(\dot{Q}_{u v1}\left(t\right)-k_{n x}\cdot Q_{u v1}\left(t\right)+D a s e\cdot\delta\left(t\right)\right.\right.}\\ &{\left.\left.\left(\dot{Q}_{u v1}\left(0\right)=0\right.\right.\right.}\\ &{\left.\left.\dot{Q}_{u v2}\left(t\right)-k_{n y\ast}\left(Q_{u v1}\right)\cdot Q_{u v2}\left(t\right)+k_{n x}\cdot Q_{u v1}\left(t\right)\right.\right.}\\ &{\left.\left.\dot{Q}_{v v2}\left(0\right)=0\right.}\\ &{\left.\dot{Q}_{v v1}\left(t\right)=-k_{n b}\cdot Q_{v v1}\left(t\right)+k_{n p t}\left(Q_{u v1}\right)\cdot Q_{u v2}\left(t\right)\right.}\\ &{\left.\left(\dot{Q}_{v v1}\left(0\right)=0\right.}\\ &{\left.\left.R a_{n d}\left(t\right)=\frac{f\cdot E_{n b}\cdot Q_{v a1}\left(t\right)}{B W}\right.\right.}\\ &{\left.\left.R a_{n d}\left(0\right)=0\right.}\end{array}
$$  

with  

$$
\begin{array}{l}{{k_{e m p t}\left(Q_{s t o}\right)=k_{m i n}+\frac{{k_{m a x}-k_{m i n}}}{2}\cdot\left\{\operatorname{tanh}\Bigl[\alpha\left(Q_{s t o}-\upbeta\cdot D o s e\right)\Bigr]\right.}}}\\ {{\left.\quad-\operatorname{tanh}\Bigl[\upbeta\left(Q_{s t o}-c\cdot D o s e\right)\Bigr]+2\Bigr\}}}\end{array}
$$  

# Endogenous Glucose Production  

$$
\begin{array}{c}{{E G P\!\left(t\right)=k_{p1}\left(t\right)-k_{p2}\cdot G_{p}\left(t\right)}}\\ {{-k_{p3}\left(t\right)\cdot X^{L}\left(t\right)+\xi\cdot X^{H}\left(t\right)}}\end{array}
$$  

$$
\dot{X}^{L}\left(t\right)=-k_{i}\cdot\left[X^{L}\left(t\right)-I^{\prime}\left(t\right)\right]\;\;\;X^{L}\left(0\right)=I_{b}
$$  

$$
\dot{I}^{\prime}(t)=-k_{i}\cdot\left[I^{\prime}(t)\!-\!I(t)\right]\;\;\;I^{\prime}(0)=I_{b}
$$  

$$
\begin{array}{r}{\dot{X}^{H}\left(t\right)=-k_{H}\cdot X^{H}\left(t\right)+k_{H}\cdot\operatorname*{max}\biggr[\left(H\left(t\right)-H_{b}\right),0\biggr]X^{H}\left(0\right)=0}\end{array}
$$  

# Glucose Utilization  

$$
U_{i i}\left(t\right)=F_{c n s}
$$  

$$
U_{i d}\left(t\right)=\frac{k_{i r}\left(t\right)\cdot\left[V_{m0}+V_{m x}\left(t\right)\cdot X\left(t\right)\cdot\left(1+r_{\mathrm{l}}\cdot r i s k\right)\right]\cdot G\left(t\right)}{K_{m0}+G_{t}\left(t\right)}
$$  

with  

$$
{\dot{X}}{\left(t\right)}=-p_{2U}\cdot X{\left(t\right)}+p_{2U}\cdot\left[I{\left(t\right)}-I_{b}\right]\;\;\;X{\left(0\right)}=0
$$  

$$
r i s k=\left\{10\atop10\cdot\left[f{\left(G\right)}\right]^{2}\right.\quad{\mathrm{if~}}G\geq G_{b}
$$  

$$
f\bigl(G\bigr)\!=\!\Bigl[\log\bigl(G\bigr)\Bigr]^{\gamma_{2}}-\Bigl[\log\bigl(G_{b}\bigr)\Bigr]^{\gamma_{2}}
$$  

Renal Excretion  

$$
E\!\left(t\right)\!=\!\left\{\!k_{e1}\!\cdot\!\!\left[G_{p}\left(t\right)\!-\!k_{e2}\right]\!\;\;\mathrm{if}\;G_{p}\left(t\right)\!>\!k_{e2}\!\right.
$$  

External Insulin Rate of Appearance  

$$
R a_{I}\left(t\right)=R a_{I s c}\left(t\right)+R a_{I i d}\left(t\right)+R a_{I i h}\left(t\right)
$$  

Subcutaneous Insulin Kinetics  

$$
R a_{I s c}(t)=k_{a1}\cdot I_{s c1}\left(t\right)+k_{a2}\cdot I_{s c2}\left(t\right)
$$  

with  

$$
\begin{array}{r l}&{\left\lceil\dot{I}_{s c1}\left(t\right)=-\big(\boldsymbol{k}_{d}+\boldsymbol{k}_{a1}\big)\cdot\boldsymbol{I}_{s c1}\left(t\right)+\boldsymbol{u}_{s c}\left(t-\tau\right)\right\rceil}\\ &{\left\lceil I_{s c1}\left(0\right)=\boldsymbol{I}_{s c1s s}}\\ &{\dot{I}_{s c2}\left(t\right)=\boldsymbol{k}_{d}\cdot\boldsymbol{I}_{s c1}\left(t\right)-\boldsymbol{k}_{a2}\cdot\boldsymbol{I}_{s c2}\left(t\right)}\\ &{\left\lceil I_{s c2}\left(0\right)=\boldsymbol{I}_{s c2s s}}\end{array}
$$  

Intradermal Insulin Kinetics  

$$
R a_{I i d}(t)=i d t_{1}(t)+k_{a}\cdot I_{i d_{2}}(t)
$$  

with  

$$
\begin{array}{r l}&{\left(\dot{I}_{i d_{1}}=-\left(0.04+k_{d}\right)\cdot I_{i d_{1}}\left(t\right)+u_{i d}\left(t\right)\right.}\\ &{\left.\left|I_{i d_{1}}\left(0\right)=I_{i d_{1}s s}\right.}\\ &{\left.\left|\dot{I}_{i d_{2}}=-k_{a}\cdot I_{i d_{2}}\left(t\right)+i d t_{2}\left({\sf t}\right)\right.}\\ &{\left.\left|I_{i d_{2}}\left(0\right)=I_{i d_{2}s s}\right.}\end{array}
$$  

where $i d t_{I}(t)$ and $i d t_{2}(t)$ are defined by the transfer functions  

$$
\begin{array}{r}{T_{1}(s)\!=\!\left(\frac{b_{\scriptscriptstyle1}}{s+b_{\scriptscriptstyle1}}\right)^{2}\!=\!\frac{\operatorname{L}\big\{i d t_{1}(t)\big\}}{\operatorname{L}\big\{0.04\cdot I_{i d_{\scriptscriptstyle1}}(t)\big\}},\:T_{2}(s)\!=\!\left(\frac{b_{\scriptscriptstyle2}}{s+b_{\scriptscriptstyle2}}\right)^{a_{\scriptscriptstyle2}}\!=\!\frac{\operatorname{L}\big\{i d t_{2}(t)\big\}}{\operatorname{L}\big\{k_{d}\cdot I_{i d_{\scriptscriptstyle1}}(t)\big\}}.}\end{array}
$$  

Inhaled Insulin Kinetics  

$$
R a_{I i h}\left(t\right)=k_{a I i h}\cdot I_{i h}\left(t\right)
$$  

with  

$$
\dot{I}_{i h}\left(t\right)=-k_{a I i h}\cdot I_{i h}\left(t\right)+F_{I i h}\cdot u_{i h}\left(t\right)
$$  

Subcutaneous Glucose Kinetics  

$$
\dot{G}_{s c}\left(t\right)=-1/T_{s}\cdot G_{s c}\left(t\right)+1/T_{s}\cdot G\!\left(t\right)\ \ G_{s c}\left(0\right)=G_{b}
$$  

Glucagon Kinetics and Secretion  

$$
\dot{H}\left(t\right)=-n\cdot H\left(t\right)+S R_{H}\left(t\right)+R a_{H}\left(t\right)\;\;\;H\left(0\right)=H_{b}
$$  

![](images/8fb5eee9439c595ea6257205db356c43a88bdbc1cc3152dcd90b585ac7449d64.jpg)  

Subcutaneous Glucagon Kinetics  

$$
\begin{array}{r l}&{\left|\dot{H}_{s c1}\left(t\right)=-\big(k_{h1}+k_{h2}\big)\!\cdot\!H_{s c1}\left(t\right)\right.}\\ &{\left.\!\int\!\!H_{s c1}\left(0\right)\!=\!H_{s c1b}}\\ &{\left.\!\!\int\!\!\dot{H}_{s c2}\left(t\right)\!=\!k_{h1}\cdot H_{s c1}\left(t\right)\!-\!k_{h3}\cdot H_{s c2}\left(t\right)\right.}\\ &{\left.\!\!\int\!\!H_{s c2}\left(0\right)\!=\!H_{s c2b}}\end{array}
$$  

$$
R a_{H}\left(t\right)=k_{h3}\cdot H_{s c2}\left(t\right)
$$  

# Abbreviations  

AP, artificial pancreas; B, breakfast; BG, blood glucose; CF, correction factor; CGM, continuous glucose monitoring; CHO, carbohydrates;  CR,  carbohydrate-to-insulin  ratio;  CSII,  continuous subcutaneous insulin infusion; CVGA, control variability grid analysis; D, dinner; EGP, endogenous glucose production; FDA, Food and Drug Administration; ID, intradermal route; IG, interstitial glucose; IH, inhaled route; L, lunch; S2008, Type 1 Diabetes Simulator 2008; S2013, Type 1 Diabetes Simulator 2013; S2017, Type 1 Diabetes Simulator 2017; SC, subcutaneous route; SD, standard deviation; SI, insulin sensitivity; SMBG, self-monitoring of blood glucose; T1D, type 1 diabetes; UVA, University of Virginia.  

# Declaration of Conflicting Interests  

The author(s) declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.  

# Funding  

The author(s) disclosed receipt of the following financial support for the research, authorship, and/or publication of this article: This study was supported by University of Padova, “Progetto di Ateneo $2014^{\circ}$ and grant CPDA145405/14, and by the Juvenile Diabetes Research Foundation grant 2-SRA-2016-291-Q-R.  

# ORCID iDs  

Roberto Visentin $\mathbb{O}$ https://orcid.org/0000-0002-5848-5990   
Enrique Campos-Náñez $\mathbb{O}$ https://orcid.org/0000-0002-3036-3772   
Claudio Cobelli $\mathbb{O}$ https://orcid.org/0000-0002-0169-6682  

# References  

1.	 Cobelli C, Dalla Man C, Sparacino G, Magni L, De Nicolao G, Kovatchev BP. Diabetes: models, signals, and control. IEEE Rev Biomed Eng. 2009;2:54-96.   
2.	 Kovatchev BP, Breton M, Dalla Man C, Cobelli C. In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes. J Diabetes Sci Technol. 2009;3(1):44-55.   
3.	 Wilinska ME, Chassin LJ, Acerini CL, Allen LM, Dunger DB, Hovorka R. Simulation environment to evaluate closedloop insulin delivery system in type 1 diabetes. J Diabetes Sci Technol. 2010;4(1):132-144. 4.	 Kanderian  SS,  Weinzimer  S,  Voskanyan  G,  Steil  GM. Identification of intraday metabolic profiles during closed-loop glucose control in individuals with type 1 diabetes. J Diabetes Sci Technol. 2009;3(5):1047-1057. 5.	 Dalla Man C, Rizza RA, Cobelli C. Meal simulation model of the glucose-insulin system. IEEE Trans Biomed Eng. 2007;54(10):1740-1749.   
6.	 Basu R, Dalla Man C, Campioni M, et al. Effects of age and sex on postprandial glucose metabolism: differences in glucose turnover, insulin secretion, insulin action, and hepatic insulin extraction. Diabetes. 2006;55(7):2001-2014. 7.	 Dalla Man C, Micheletto F, Lv D, Breton M, Kovatchev BP, Cobelli C. The UVA/PADOVA Type 1 Diabetes Simulator: new features. J Diabetes Sci Technol. 2014;8(1):26-34.   
8.	 Visentin R, Dalla Man C, Kovatchev BP, Cobelli C. The University of Virginia/Padova Type 1 Diabetes Simulator matches the glucose traces of a clinical trial. Diabetes Technol Ther. 2014;16(7):428-434. 9.	 Anderson SM, Raghinaru D, Pinsker JE, et al. Multinational home use of closed-loop control is safe and effective. Diabetes Care. 2016;39(7):1143-1150.   
10.	 Brown SA, Breton MD, Anderson SM, et al. Overnight closedloop control improves glycemic control in a multicenter study of adults with type 1 diabetes. J Clin Endocrinol Metab. 2017;102(10):3674-3682   
11.	 Kropff J, Del Favero S, Place J, et al. 2 month evening and night closed-loop glucose control in patients with type 1 diabetes under free-living conditions: a randomised crossover trial. Lancet Diabetes Endocrinol. 2015;3(12):939-947.   
12.	 Renard E, Farret A, Kropff J, et al. Day-and-night closed-loop glucose control in patients with type 1 diabetes under freeliving conditions: results of a single-arm 1-month experience compared with a previously reported feasibility study of evening and night at home. Diabetes Care. 2016;39(7):1151-1160.   
13.	 Kovatchev B, Cheng P, Anderson SM, et  al. feasibility of long-term closed-loop control: a multicenter 6-month trial of 24/7 automated insulin delivery. Diabetes Technol Ther. 2017;19(1):18-24.   
14.	 Dassau E, Brown SA, Basu A, et  al. Adjustment of openloop settings to improve closed-loop results in type 1 diabetes: a multicenter randomized trial. J Clin Endocrinol Metab. 2015;100(10):3878-3886.   
15.	 Pinsker JE, Lee JB, Dassau E, et  al. Randomized crossover comparison of personalized MPC and PID control algorithms for the artificial pancreas. Diabetes Care. 2016;39(7):1135- 1142.   
16.	 Forlenza GP, Deshpande S, Ly TT, et al. Application of zone model predictive control artificial pancreas during extended use of infusion set and sensor: a randomized crossover-controlled home-use trial. Diabetes Care. 2017;40(8):1096-1102.   
17.	 Owens C, Zisser H, Jovanovic L, Srinivasan B, Bonvin D, Doyle FJ III. Run-to-run control of blood glucose concentrations for people with type 1 diabetes mellitus. IEEE Trans Biomed Eng. 2006;53(6):996-1005.   
18.	 Magni L, Forgione M, Toffanin C, et al. Run-to-run tuning of model predictive control for type 1 diabetes subjects: in silico trial. J Diabetes Sci Technol. 2009;3(5):1091-1098.   
19.	 Wang Y, Dassau E, Doyle FJ III. Closed-loop control of artificial pancreatic beta-cell in type 1 diabetes mellitus using model predictive iterative learning control. IEEE Trans Biomed Eng. 2010;57(2):211-219.   
20.	 Toffanin C, Visentin R, Messori M, Di Palma F, Magni L, Cobelli C. Towards a run-to-run adaptive artificial pancreas: in silico results [published online ahead of print January 11, 2017]. IEEE Trans Biomed Eng. doi:10.1109/ TBME.2017.2652062.   
21.	 Messori M, Kropff J, Del Favero S, et al. Individually adaptive artificial pancreas improves glucose control in subjects with type 1 diabetes. a one-month free-living conditions trial. Diabetes Technol Ther. 2017;19(10):560-571. doi:10.1089/ dia.2016.0463.   
22.	 Reddy M, Pesl P, Xenou M, et al. Clinical safety and feasibility of the advanced bolus calculator for type 1 diabetes based on case-based reasoning: a 6-week nonrandomized single-arm pilot study. Diabetes Technol Ther. 2016;18(8):487-493.   
23.	 Haidar A, Wilinska ME, Graveston JA, Hovorka R. Stochastic virtual population of subjects with type 1 diabetes for the assessment of closed loop glucose controllers. IEEE Trans Biomed Eng. 2013;60(12):3524-3533.   
24.	 Herrero P, Pesl P, Bondia J, et al. Method for automatic adjustment of an insulin bolus calculator: in silico robustness evaluation under intra-day variability. Comput Methods Programs Biomed. 2015;119(1):1-8.   
25.	 Herrero P, Bondia J, Adewuyi O, et  al. Enhancing automatic closed-loop glucose control in type 1 diabetes with an adaptive meal bolus calculator—in silico evaluation under intra-day variability. Comput Methods Programs Biomed. 2017;146:125-131.   
26.	 Visentin R, Dalla Man C, Cobelli C. One-day Bayesian cloning of type 1 diabetes subjects: toward a single-day UVA/ Padova Type 1 Diabetes Simulator. IEEE Trans Biomed Eng. 2016;63(11):2416-2424.   
27.	 Hinshaw L, Dalla Man C, Nandy DK, et al. Diurnal pattern of insulin action in type 1 diabetes implications for a closed-loop system. Diabetes. 2013;62(7):2223-2229.   
28.	 Visentin R, Dalla Man C, Kudva YC, Basu A, Cobelli C. Circadian  variability  of  insulin  sensitivity:  physiological input for in silico artificial pancreas. Diabetes Technol Ther. 2015;17(1):1-7.   
29.	 Mallad A, Hinshaw L, Dalla Man C, et al. Nocturnal glucose metabolism in type 1 diabetes: a study comparing single versus dual tracer approaches. Diabetes Technol Ther. 2015;17(8):587- 595.   
30.	 Perriello G, De Feo P, Torlone E, et al. The dawn phenomenon in type 1 (insulin-dependent) diabetes mellitus: magnitude, frequency, variability, and dependency on glucose counterregulation and insulin sensitivity. Diabetologia. 1991;34(1):21-28.   
31.	 Hinshaw L, Mallad A, Dalla Man C, et  al. Glucagon sensitivity and clearance in type 1 diabetes: insights from in vivo and in silico experiments. Am J Physiol Endocrinol Metab. 2015;309(5):E474-E486.   
32.	 Schiavon M, Dalla Man C, Cobelli C. Modeling subcutaneous absorption of fast-acting insulin in type 1 diabetes [ published online ahead of print December 15, 2017]. IEEE Trans Biomed Eng. doi:10.1109/TBME.2017.2784101.   
33.	 Lv D, Kulkarni SD, Chan A, et  al. Pharmacokinetic model of the transport of fast-acting insulin from the subcutaneous and intradermal spaces to blood. J Diabetes Sci Technol. 2015;9(4):831-840.   
34.	 Visentin R, Giegerich C, Jäger R, et al. Improving efficacy of inhaled technosphere insulin (Afrezza) by postmeal dosing: in silico clinical trial with the UVA/Padova Type 1 Diabetes Simulator. Diabetes Technol Ther. 2016;18(9):574-585.   
35.	 Vettoretti M, Facchinetti A, Sparacino G, Cobelli C. A model of self-monitoring blood glucose measurement error. J Diabetes Sci Technol. 2017;11(4):724-735.   
36.	 Vettoretti M, Facchinetti A, Sparacino G, Cobelli C. Type 1 diabetes patient decision simulator for in silico testing safety and  effectiveness  of  insulin  treatments  [published  online ahead of print August 29, 2017]. IEEE Trans Biomed Eng. doi:10.1109/TBME.2017.2746340.   
37.	 Facchinetti A, Del Favero S, Sparacino G, Cobelli C. Model of glucose sensor error components: identification and assessment for new Dexcom G4 generation devices. Med Biol Eng Comput. 2015;53(12):1259-1269.   
38.	 Campos-Náñez E, Fortwaengler K, Breton MD. Clinical impact of blood glucose monitoring accuracy: an in-silico study [published online ahead of print May 1, 2017]. J Diabetes Sci Technol. doi:10.1177/1932296817710474.   
39.	 Davidson PC, Hebblewhite HR, Steed RD, Bode BW. Analysis of guidelines for basal-bolus insulin dosing: basal insulin, correction factor, and carbohydrate-to-insulin ratio. Endocr Pract. 2008;14(9):1095-1101.   
40.	 Haidar A, Wilinska ME, Graveston JA, Hovorka R. Stochastic virtual population of subjects with type 1 diabetes for the assessment of closed-loop glucose controllers. IEEE Trans Biomed Eng. 2013;60(12):3524-3533.   
41.	 Schiavon M, Persson M, Dalla Man C, et  al. Novel tracer approach  to  estimate  postprandial  complex  carbohydrate metabolism [abstract]. Diabetologia. 2016;59(suppl 1):s268.   
42.	 Schiavon M, Visentin R, Dalla Man C, Klabunde T, Cobelli C. Modeling subcutaneous absorption of U100 and U300 insulin glargine in type 1 diabetes [abstract]. Diabetes Technol Ther. 2017;19(suppl 1):A118.  