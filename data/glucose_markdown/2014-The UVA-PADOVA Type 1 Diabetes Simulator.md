# The UVA/PADOVA Type 1 Diabetes Simulator: New Features  

Chiara Dalla Man, PhD1, Francesco Micheletto, PhD1, Dayu Lv, PhD2, Marc Breton, PhD2, Boris Kovatchev, PhD2, and Claudio Cobelli, PhD  

# Abstract  

Recent studies have provided new insights into nonlinearities of insulin action in the hypoglycemic range and into glucagon kinetics as it relates to response to hypoglycemia. Based on these data, we developed a new version of the UVA/PADOVA Type 1 Diabetes Simulator, which was submitted to FDA in 2013 (S2013). The model of glucose kinetics in hypoglycemia has been improved, implementing the notion that insulin-dependent utilization increases nonlinearly when glucose decreases below a certain threshold. In addition, glucagon kinetics and secretion and action models have been incorporated into the simulator: glucagon kinetics is a single compartment; glucagon secretion is controlled by plasma insulin, plasma glucose below a certain threshold, and glucose rate of change; and plasma glucagon stimulates with some delay endogenous glucose production. A refined statistical strategy for virtual patient generation has been adopted as well. Finally, new rules for determining insulin to carbs ratio (CR) and correction factor (CF) of the virtual patients have been implemented to better comply with clinical definitions. S2013 shows a better performance in describing hypoglycemic events. In addition, the new virtual subjects span well the real type 1 diabetes mellitus population as demonstrated by good agreement between real and simulated distribution of patient-specific parameters, such as CR and CF. S2013 provides a more reliable framework for in silico trials, for testing glucose sensors and insulin augmented pump prediction methods, and for closed-loop single/dual hormone controller design, testing, and validation.  

# Keywords  

computer simulation, diabetes control, modeling  

The development of artificial pancreas control algorithms in the past few years has been accelerated by computer simulation; for example, in silico testing has provided directions for clinical studies and ruling out of ineffective control scenarios in a cost-effective manner.1,2 To the best of our knowledge the first example of a closed-loop system for an artificial pancreas was designed in silico and used in the ADICOL project.3 However, the extensive use of computer simulation started in 2008 when a paradigm change occurred in the field of type 1 diabetes mellitus (T1DM): for the first time a computer model has been accepted by FDA as a substitute for preclinical trials of certain insulin treatments, including closed-loop  algorithms.4  This  UVA-PADOVA  Type  1 Diabetes Simulator (S2008) emulated meal challenges and included a population of 300 in silico subjects (100 adults, 100 adolescents, 100 children). Each virtual subject was represented by a model parameter vector, which was randomly extracted from an appropriate joint parameter distribution. The S2008 has been successfully used by 32 research groups in academia, as well as by companies active in the field of T1DM; simulation results were presented by 63 publications in peer-reviewed journals.  

Recently new data and models have become available on hypoglycemia5 and counterregulation.6,7 This put us in the position to add new features and models leading to a new degree of accuracy of this important simulation tool. To further increase the simulator quality, new rules for determining CSII parameters of the virtual patients have been implemented and a new statistical strategy for virtual patient generation has been adopted. This article presents the new features of the UVA-PADOVA Type 1 Diabetes Simulator, version S2013, submitted to FDA in April 2013.  

# Method  

# T1DM Simulator 2008 (S2008)  

The Model.  The model describing the glucose-insulin system during a meal is described in detail in Kovatchev et al.4 Its schematic representation is shown in Figure 1 (white and gray blocks). Briefly, the model puts in relation plasma concentrations, that is, glucose $G$ and insulin $I_{\leq}$ , with glucose fluxes, that is, endogenous glucose production $(E G P)$ , glucose rate of appearance $(R a)$ , glucose utilization $(U)$ , renal extraction $(E)$ , and insulin fluxes, that is, rate of insulin appearance from the subcutaneous tissue $(S C)$ and insulin degradation $(D)$ . The glucose subsystem consists of a 2-compartment model: insulin-independent utilization occurs in the first compartment, representing plasma and rapidly equilibrating tissues, while insulin-dependent utilization occurs in the second compartment, representing slowly equilibrating tissues. The insulin subsystem is also described with 2 compartments, representing liver and plasma, respectively. Subcutaneous insulin kinetics is represented by a 3-compartment model. Endogenous glucose production is assumed to be linearly dependent on plasma glucose concentration and a delayed  insulin  signal.  Glucose  rate  of  appearance  is described with a model of glucose transit through the stomach and intestine, with the stomach represented by 2 compartments, while a single compartment is used to describe the gut; the rate constant of gastric emptying is a nonlinear function of the amount of carbohydrates in the stomach. Glucose utilization during a meal has 2 components: insulinindependent utilization by the brain and the erythrocytes takes place in the first compartment and is constant, while insulin-dependent utilization takes place in the remote compartment and depends nonlinearly from glucose in the tissues.  Here  we  report  the  equations  describing insulin-dependent utilization, $U_{i d}(t)$ , and endogenous glucose production, $E G P(t)$ , of S2008, to evidence the modifications incorporated in S2013, described in the following section.  

The model of glucose kinetics is described by,  

$$
\begin{array}{r}{\left\{\dot{G}_{p}=E G P-U_{i i}-k_{1}\cdot G_{p}\left(t\right)+k_{2}\cdot G_{t}\left(t\right)\ \ G_{p}\left(0\right)=G_{p b}\ }\\ {\dot{G}_{t}=-U_{i d}\left(t\right)+k_{1}\cdot G_{p}\left(t\right)-k_{2}\cdot G_{t}\left(t\right)\ \ \ \ G_{t}\left(0\right)=G_{p b}\,\frac{k_{1}}{k_{2}}}\end{array}
$$  

with $G_{p}(t)$ amount of glucose in plasma, $G_{\it t}(t)$ amount of glucose in the tissue, and $k_{\mathnormal{1}}$ and $k_{2}$ rate parameters. The insulinindependent utilization by the brain and the erythrocytes $(U_{i i})$ takes place in the first compartment and is constant. The insulin-dependent utilization $(U_{i d})$ takes place in the remote compartment and depends from glucose in the tissues by a Michaelis–Menten relationship,  

$$
U_{i d}\left(t\right)=\frac{\left[V_{m0}+V_{m x}\cdot X(t)\right]\cdot G_{t}\left(t\right)}{K_{m0}+G_{t}\left(t\right)}
$$  

$$
\dot{X}(t)=-p_{2U}\cdot X(t)+p_{2U}\cdot\left[I(t)-I_{b}\right]\;\;\;X(0)=0
$$  

with $X(t)$ insulin action on glucose utilization and $V_{m\o j}\ {\cal V}_{m\o j}$ $K_{_{m\theta}},p_{_{2U}}$ rate parameters.  

The EGP model is,  

$$
\begin{array}{c}{{E G P(t)=k_{p1}-k_{p2}\cdot G_{p}(t)-k_{p3}\cdot X^{L}(t)}}\\ {{\dot{I}\left\langle t\right\rangle=-k_{i}\cdot\left[I\left\langle t\right\rangle-I(t)\right]\ \ \ \ I\left\langle0\right\rangle=I_{b}}}\\ {{\dot{X}^{L}(t)=-k_{i}\cdot\left[X^{L}(t)-I\left\langle t\right\rangle\right]\ \ \ \ X^{L}(0)=I_{b}}}\end{array}
$$  

with $G_{p}(t)$ glucose amount in the plasma compartment, $X^{L}(t)$ delayed insulin action in the liver, and $k_{i},\;k_{p l},\;k_{p2},\;k_{p3}$ rate parameters.  

T1DM Virtual Subjects.  The S2008 is equipped with 100 virtual adults, 100 adolescents and 100 children. These T1DM populations have been generated by randomly extracting, from joint parameter distributions, different realizations of the parameter vector, that is, the vector that includes the whole parameter set of the model. At the time of the first release of the simulator, reliable joint parameter distributions were available for a nondiabetic adult population.4 Thus, the T1DM parameter joint distributions were derived from those available in the nondiabetic adult populations. In particular, the intersubject variability was assumed to be the same (same covariance matrix), but certain clinically relevant modifications were introduced in the average parameter vector. For instance, basal endogenous glucose production was assumed to be higher in T1DM subjects. Similarly, parameter distributions in children and adolescent were obtained from T1DM adults by introducing some changes in the average parameter vector, for example, insulin sensitivity was assumed higher in children and lower in adolescents compared to adults.4  

# T1DM Simulator 2013 (S2013)  

S2013 presents several important improvements with respect to S2008, both on the model on which the simulator is based, but also on the joint parameter distribution, the definition of clinically relevant parameters, and the strategy for virtual patient generation.  

New Model.  The scheme of the glucose model of the S2013 is shown in Figure 1: white blocks represent the unit processes that are the same of S2008 (gastro-intestinal tract, glucose and insulin kinetics), gray blocks have been updated to account for counterregulation (liver, muscle, and adipose tissue), and black blocks are new (alpha cell, glucagon kinetics, and delivery). Specifically, the model of Kovatchev et al4 was modified as follows (see appendix for the complete set of model equations).  

![](images/4867495f76de8c7ce15fb574ef30b0bf7e4f136617e11c960f4591cd4f001abc.jpg)  
Figure 1.  Scheme of the model included in the FDA-accepted T1DM simulator. White blocks are the unit processes of $\dot{\varsigma}2008^{4}$ (gastro-intestinal tract, glucose kinetics and insulin kinetics); gray blocks are those that have been updated in the S2013 to account for counterregulation (liver, muscle, and adipose tissue); black blocks are new (alpha cell, glucagon kinetics, and delivery).  

Glucagon secretion and kinetics.  Glucagon kinetics are described with a 1-compartment linear model8:  

$$
\dot{H}(t)=-n\cdot H(t)+S R_{H}(t)\;\;\;H(0)=H_{b}
$$  

with $H(t)$ plasma hormone concentration, $S R_{_H}(t)$ glucagon secretion $\boldsymbol{S}\boldsymbol{R}_{H}^{b}$ its basal value), and $n$ clearance rate.  

Glucagon  secretion  is  described  as  the  sum  of  2 components:  

$$
S R_{H}(t)=S R_{H}^{s}(t)+S R_{H}^{d}(t)
$$  

with  

$$
\dot{S}R_{H}^{s}(t)=\left\{\begin{array}{c}{\displaystyle-\mathsf{p}\cdot\left[S R_{H}^{s}(t)-m a x\left(\mathbb{\sigma}_{2}\cdot\left[G t h-G(t)\right]+S R_{H}^{b},0\right)\right]i f\,G(t)\geq G_{b}}\\ {\displaystyle-\mathsf{p}\cdot\left[S R_{H}^{s}(t)-m a x\left(\frac{\sigma\cdot\left[G t h-G(t)\right]}{I(t)+I}+S R_{H}^{b},0\right)\right]f\,G(t)<G_{b}}\end{array}\right.
$$  

with $G(t)$ plasma glucose ( $G_{b}$ its basal value), $I(t)$ plasma insulin concentration $\boldsymbol{I}_{b}$ its basal value), $\sigma$ and $\boldsymbol{\upsigma}_{2}$ alpha-cell responsivity to glucose level, $I/\rho$ delay between static glucagon secretion and plasma glucose. In this way, static secretion is stimulated when $G{<}G_{_b}$ (but modulated by insulin) and inhibited when $G\!\!\geq\!G_{_{b}}$ .  

The second component is proportional to glucose rate of change,  

$$
\mathrm{SR}_{\mathrm{H}}^{\mathrm{~d~}}(t)=\delta\cdot\operatorname*{max}\left(-\frac{d G(t)}{d t},0\right)
$$  

with dG(t)/dt glucose rate of change, and $\delta$ alpha-cell responsivity to glucose rate of change.  

Model parameters $\upsigma$ and δ also reflect that glucagon response declines with age of diabetes (see also the “New Joint Parameter Distribution” section below).  

Of note, in real life, glucagon secretion is almost certainly dependent on insulin level in the alpha cells (paracrine effect), not in the circulation. However, it is very difficult to model the intrapancreatic levels, so the use of plasma insulin was the best we could do, even if not perfectly physiologic.  

Glucagon action.  The model of glucagon action is based on the EGP model reported in equation (4), but assumes that EGP is stimulated by the above basal glucagon concentration with some delay,  

$$
\begin{array}{r l}&{E G P(t)=k_{p1}-k_{p2}\cdot G_{p}\left(t\right)-k_{p3}\cdot X^{L}(t)+\xi\cdot X^{H}(t)}\\ &{\qquad\qquad i\left.\uparrow(t)=-k_{i}\cdot\left[I\left(t\right)-I(t)\right]\qquad I\left[0\right)=I_{b}}\\ &{\dot{X}^{L}(t)=-k_{i}\cdot\left[X^{L}(t)-I\left(t\right)\right]\qquad X^{L}(0)=I_{b}}\\ &{\dot{X}^{H}(t)=-k_{H}\cdot X^{H}(t)+k_{H}\cdot\operatorname*{max}\left[\left(H(t)-H_{b}\right),0\right]}\\ &{X^{H}(0)=0}\end{array}
$$  

with $H(t)$ plasma glucagon concentration, $X^{H}(t)$ delayed glucagon action on EGP, $\boldsymbol{\xi}$ liver responsivity to glucagon, and $1/k_{_H}$ delay  between  glucagon  concentration  and action.  

Glucose utilization in hypoglycemia.  The model of glucose utilization reported in equations (2) and (3) is unable to describe well the hypoglycemic range likely due to an inadequate description of insulin action, which paradoxically increases when glucose decreases under a given threshold. This phenomenon has been described during hyperinsulemic clamps in T1DM.9,10 Based on these observation we developed a new model that assumes that insulin-dependent utilization $U_{i d}(t)$ increases when glucose decreases below a certain threshold, following the blood glucose risk function:11  

$$
U i d(t)=\frac{\Big[V_{m0}+V_{m x}\cdot X(t)\cdot\big(1+r_{1}\cdot r i s k\big)\Big]\cdot G_{t}(t)}{K_{m0}+G_{t}(t)}
$$  

where  

$$
r i s k=\left\{10\atop{\left.\right|10\cdot\left[f(G)\right]^{2}}\right.\quad\mathrm{if~G_{\th}\leq G<G_{\ b}~}
$$  

with $G_{b}$ basal glucose, $G_{_{t h}}$ the hypoglycemic threshold (set at $60\,\mathrm{mg/dl})$ ),  

$$
f(G)=\log\left({\frac{G}{G_{b}}}\right)^{r_{2}}
$$  

and $r_{\scriptscriptstyle I^{\prime}}\,r_{\scriptscriptstyle2}$ model parameters.  

The model was validated on 32 T1DM (age $38\pm12$ years, height $174\pm10\ \mathrm{cm}$ , weight $78\pm12\mathrm{\,kg})$ ), which underwent a hyperinsulinemic  euglycemic  and  hypoglycemic  clamp, where hypoglycemia is induced at a descending rate of $1~\mathrm{mg}/$ $\mathrm{dl}/\mathrm{min}$ until $50\,\mathrm{mg/dl}$ glucose value is reached. However, the protocol did not include tracer administration, which would have enabled the estimate of endogenous glucose production and thus the use of the system decomposition and forcing function strategy. Thus the only model capable of describing the data is the glucose minimal model,12 which couples insulin action on glucose utilization and production,  

![](images/706bca7977edfd30565317a2c2693c9e2551d12bd6774913fd57c5271ee87e79.jpg)  
Figure 2.  Average model prediction against glucose data in 32 T1DM subjects (unpublished). The gray area represents the counterregulation needed to return in normal glycemia (vertical bars are standard deviations).  

counterregulation response to hypoglycemia, it was expected that, to be valid, the prediction of the model for $t\geq\hat{t}_{*}$ , would underestimate the glucose data. Figure 2 shows the results: the model fits well blood glucose data for $0\leq t<\hat{t}$ indeed and, as expected, underestimates the glucose data for $t\geq\hat{t}$ .  

Subcutaneous glucagon transport.  As the employment of subcutaneous glucagon in a closed-loop system has received more attention (see, eg, El-Khatib et al),13 the simulation platform necessitates including a pharmacokinetic model of exogenous glucagon to reproduce its appearance in plasma, its action controlled by the model described previously.  

S2013 incorporates a recently proposed model, which describes subcutaneous glucagon transit with a 2-compartment model,14  

$$
\dot{G}(t)=\left\{\begin{array}{l l}{-[S G+X(t)\cdot(1+r_{1}\cdot r i s k)]\cdot G(t)+S G\cdot G_{b}}&\\ {\quad+\displaystyle{\frac{G I R(t)}{V}}i f\,G(t)<G_{b}}&\\ {\quad-[S G+X(t)]\cdot G(t)+S G\cdot G_{b}+\displaystyle{\frac{G I R(t)}{V}}}&\\ {G(0)=G_{b}}&{G(0)=G_{b}}&\\ {o t h e r w i s e}&\\ {\dot{X}(t)=-p_{2}\cdot X(t)+p_{3}\cdot S I\cdot[I(t)-I_{b}]}&{X(0)=0}\end{array}\right.
$$  

$$
\begin{array}{r l}&{\left|\dot{H}_{s c1}(t)=-\big(k_{h1}+k_{h2}\big)\!\cdot\!H_{s c1}(t)+H_{\mathrm{inf}}\left(t\right)\right.}\\ &{\left.\!\int\!\!H_{s c1}(0)=H_{s c1b}\,}\\ &{\left|\dot{H}_{s c2}(t)=k_{h1}\cdot H_{s c1}(t)\!-\!k_{h3}\cdot H_{s c2}(t)\right.}\\ &{\left.\!\int\!\!H_{s c2}(0)=H_{s c2b}\,}\end{array}
$$  

where $G(t)$ is plasma glucose concentration, $I(t)$ is plasma insulin concentration, $G I R(t)$ is exogenous glucose infusion rate, SG, SI, and $p_{2}$ are model parameters, and risk is defined as in (11-12).  

For model identification, a glucose threshold of $60~\mathrm{mg/dl}$ was defined together with its corresponding time $\hat{t}$ . The model was identified on blood glucose data for time $0\leq\hat{t}<\hat{t}_{1}$ , while for $t\geq\hat{t},$ , the model was used to predict the glucose data. Since this model does not take into account the  

$$
R a_{H}(t)=k_{h3}\cdot H_{s c2}(t)
$$  

where $k_{_{h i}}$ are rate parameters describing subcutaneous glucagon kinetics, $H_{_{i n f}}$ is the glucagon infusion rate, and $H_{_{s c l}}$ and $H_{_{s c\underline{{{2}}}}}$ are glucagon concentrations in the subcutaneous space.  

This model was assessed on a clinical data set kindly provided by Dr Damiano’s group (Boston University), which was collected in a bihormonal closed-loop clinical trial.15 The data consist of 11 adults with T1DM (age: $40\pm16$ years, weight: $83\pm13\,\mathrm{kg},$ , BMI: $28\pm3\;\mathrm{kg}/\mathrm{m}^{2}$ , diabetes duration: 23 $\pm\,13$ years; and HbA1c: $7.3\pm0.8\%$ ). The patients were studied for 27 hours, during which they were given 3 regular carbohydrate-rich meals. Some patients enrolled twice and were studied with a separation of at least 5 months, leading to 13 data sets available for analysis.  

We acknowledge that the way this model was built suffers some important limitations. In particular, it was hard to isolate the effect of insulin on the response to glucagon since insulin is constantly changing. Furthermore, glucagon is given repeatedly in this type of study. However, these data were helpful to developing a quantitative model of subcutaneous glucagon kinetics.  

New Joint Parameter Distribution.  The inclusion of the risk function to the description of insulin dependent utilization makes the choice of the basal glucose $(G_{_b})$ potentially critical: insulin sensitivity starts increasing, following the risk function, as soon as glucose falls below the basal value. In S2008, $G_{b}$ was randomly generated from the joint distribution with an average of $140\,\mathrm{mg/dl}$ (chosen to reflect the knowledge available to the authors at that time). However, $G_{b}$ should correspond to patient glucose target, since it is the glucose level reached with optimal basal insulin infusion, in absence of external perturbation, like meals and physical activity. In our recent closed-loop T1DM clinical trials glucose target was around $120~\mathrm{mg/dl}$ on average. Thus, in S2013, $G_{b}$ was randomly generated from the joint distribution with an average of $120\;\mathrm{mg/dl}$ . For what concerns the new parameters of the model $(r_{_{I}},\,r_{_{2}},\,H_{_{b}},\,n,\,\xi,\,\rho,$ , and $k_{_H})$ , they were randomly generated from an appropriate joint distribution. The generation of the glucagon secretion parameters (σ and δ) was more complex as it has been shown that glucagon secretion is dependent on the duration of diabetes;15-17 that is, counterregulatory response wanes with duration of diabetes15 following a logarithmic relationship. The in silico population of S2008 did not contain duration of T1DM but only age. Thus, we needed to associate duration of T1DM to each in silico subject in a way that it was compatible with model parameters related to glucagon secretion. Using literature on T1DM incidence,18-22 we generated a distribution of age at onset (which then can be easily transformed in T1DM duration by using the subject’s age). Random sampling form this distribution, conditioned by age of in silico patient, allowed the generation of the duration of T1DM for each in silico subject, while respecting the incidence characteristics of the disease. The choice to include glucagon secretion was based on our recent data in $\mathrm{TlDM}^{23}$ showing a nonnegligible glucagon secretion. However, since not all the authors report the same slow decline in glucagon secretion found in Lorenzi et al15 (see, eg, Siafarikas et al),16 future versions of the simulator may include the possibility for the user to define the residual glucagon secretion in the population.  

Determination of CR and CF.  In real patients, CR and CF are empirically determined from patient history/habits and physician experience. At variance with S2008, where some theoretical definition of CR and CF were implemented, here we used the following definitions, which mimic as much as possible the criterion used to empirically determine them in real patients.24-27  

CR was determined with the following simulation. Each subject receives $50~\mathrm{g}$ of CHO, starting from his basal level. The optimal insulin bolus is determined so that (1) glucose concentration, measured 3 hours after the meal, is between $85\%$ and $110\%$ of the basal; (2) the minimum glucose concentration is above $90~\mathrm{mg/dl}$ ; and (3) the maximum glucose concentration is between 40 and $80~\mathrm{mg/dl}$ above the basal level. CR is then calculated as the ratio between the amount of ingested CHO and the optimal insulin bolus:  

$$
C R=\frac{i n g e s t e d\ C H O}{o p t i m a l\ b o l u s}
$$  

CF was determined with the so-called 1700 rule,27 that is,  

$$
C F=\frac{1700}{T D I}
$$  

where $T D I$ is the total daily insulin, determined for each virtual patient, using optimal CR and basal infusion rate, and assuming an average diet of $180~\mathrm{g}$ of CHO for adolescents and adults and $135\mathrm{~g~}$ for children.  

New Virtual Subject Generation.  In S2008, the virtual subjects were randomly extracted from a given joint parameter distribution. However, due to the randomness of the generation, the procedure may potentially produce physiologically nonplausible parameters in some in silico patient.  

Thus, in the S2013, we introduced new criteria for virtual subject generation. In particular, only subjects who met the following criteria have been included in the new in silico population: (1) $\mathrm{CR}\leq30~\mathrm{g/U}$ for adult and adolescents and $\mathrm{CR}\leq40~\mathrm{g/U}$ for children; (2) steady state glucose in absence of insulin infusion $>300~\mathrm{mg/dl}$ ; and (3) Mahalanobis distance lower than that corresponding to the $95\%$ percentile.  

# Simulation Environment  

The S2013 is implemented in MATLAB $\textsuperscript{\textregistered}$ R2012. The simulation environment is similar to that of S2008. In particular, the main user interface window of the software (Figure 3) allows one (1) to define a testing scenario, that is, a schedule of meals with corresponding CHO amounts; (2) to select subjects for running the desired scenario; (3) to select the hardware, for example, a particular glucose sensor and insulin pump; (4) to select a set of outcome metrics, for example, average glycemia, temporal glucose variability, and associated risks for hypoglycemia and hyperglycemia. Several graphical results are also available, for example, glucose traces, individual Poincaré plots of glucose dynamics, and control variability grid analysis (CVGA).28  

# Simulation Results  

One hundred in silico adults, adolescent, and children have been generated using the above described criteria. Average and range (minimum and maximum) of the main metabolic parameters in the 3 groups are reported in Table 1.  

Table 1.  Key Demographic and Metabolic Parameters of the In Silico Subjects Available in the Simulation Environment.   

![](images/63bd5df05697784218fcb7edc24f59393ef2282a5452a0861d7dd0f47ac7f88d.jpg)  

![](images/872525e81fc4d3acf4c391f436535ac39d2f011c4d04c770891dd620746d4988.jpg)  
Figure 3.  The user interface window of S2013.  

The distributions of in silico CR and CF in the 3 populations are shown in Figure 4. They well reproduce the distribution observed in real patients. In particular, CFs in children are higher than in adolescents and adults, reflecting both the higher insulin sensitivity and the lower body weight.  

Plasma glucose, insulin, and glucagon concentration profiles of the 100 in silico adults, adolescent and children receiving $50~\mathrm{g}$ of carbs at $8{\mathrm{:}}00~{\mathrm{AM}}$ and the optimal insulin bolus (according to patient own CR) are shown in Figure 5. Glucose variability is larger in children and adolescents than in adults, reflecting the knowledge that diabetes is more difficult to control in these cohorts. The higher glucose excursion observed in children can be explained with the fact that they received a greater amount of CHO per kilogram of body weight compared to adults. As a matter of fact, when children, adolescents, and adults received the same amount of carbs in proportion to their body weight $(1{\mathrm{~g/kg}})$ , glucose concentrations were similar in the 3 groups (data not shown).  

![](images/9d45aed2144ead1a5e9a80586effd8997f23a74c1f67b116a8d72912d4c3d6bb.jpg)  
Figure 4.  Simulated distribution of insulin to carbs ratio (CR, left) and correction factor (CF, right panels), in adult (upper), adolescent (middle), and children (lower panels) populations.  

# Conclusions  

S2013 has been described focusing on the new features introduced with respect to the S2008.4 In particular, a new module describing glucose kinetics in hypoglycemia has been added. This also required the addition of a module describing counterregulation, that is, glucagon kinetics, secretion, and action, which is essential, first, to describe well hypoglycemic phenomena and, second, in view of possibly testing dual hormone control algorithms.  

S2013 has been also improved in the virtual patient generation and characterization. In particular, we introduced a more realistic definition of important clinical parameters, such as insulin to carbs ratio and correction factor. Then, we implemented new criteria for virtual subject generation to automatically discard subjects with a nonplausible physiological behavior.  

![](images/d02ac9bf191eba197e76f1683c35e385a042c74a0da516e57f5bbd899c42b3da.jpg)  
Figure 5.  Simulated plasma glucose (upper), insulin (middle), and glucagon (lower panels) in the 100 in silico adults (left), adolescents (middle), and children (right panels).  

Simulation results have been presented for single meal optimal open-loop therapy in adult, adolescent, and children populations. S2013 provides a more reliable framework for in silico trials for regulatory purposes, for testing glucose sensors and insulin augmented pump prediction methods, and for closed-loop single/dual hormone controller design, testing, and validation. However, it is worth noting that both S2008 and S2013 simulators have been validated and accepted by FDA for a single meal scenario only. Multiple meal scenarios can obviously be simulated, but since the simulator does not include time-varying parameters, the results would not be realistic. Inclusion of meal-by-meal and day-by-day parameter variations in the simulator is under investigation. We anticipate that the first step will be the incorporation of insulin sensitivity meal-by-meal variation as reported in Hinshaw et al.23  

# Appendix  

Model Equations  

Glucose subsystem:  

$$
\left\{\begin{array}{l l}{\dot{G}_{p}(t)=E G P(t)+R a(t)-U_{i}(t)-E(t)-k_{1}\cdot G_{p}(t)+k_{2}\cdot G_{t}(t)}\\ {G_{p}(0)=G_{p b}}\\ {\dot{G}_{t}(t)=-U_{i d}(t)+k_{1}\cdot G_{p}(t)-k_{2}\cdot G_{t}(t)}\\ {G_{t}(0)=G_{b}}\\ {G(t)=\displaystyle\frac{G_{p}}{V_{G}}}\\ {G(0)=G_{b}}\end{array}\right.
$$  

Insulin subsystem:  

$$
\begin{array}{l}{\displaystyle\left\[\dot{I}_{p}(t)=-(m_{2}+m_{4})\cdot I_{p}(t)+m_{1}\cdot I_{l}(t)+R_{a i}(t)\right.}\\ {\displaystyle I_{p}(0)=I_{p b}}\\ {\displaystyle\dot{I}_{l}(t)=-(m_{1}+m_{3})\cdot I_{l}(t)+m_{2}\cdot I_{p}(t)\ I_{p l}(0)=I_{l b}}\\ {\displaystyle I(t)=\frac{I_{p}(t)}{V_{I}}}\end{array}
$$  

Glucose rate of appearance:  

$$
\begin{array}{r l}&{\left|\mathcal{Q}_{m}(t)=\mathcal{Q}_{m+2}(t)+\mathcal{Q}_{m2}(t)\right.}\\ &{\left.\left|\mathcal{Q}_{m}(0)=0\right.\right.}\\ &{\left.\left|\mathcal{Q}_{m+1}(t)-\mathcal{L}_{g_{m}}(t)+D\cdot\delta(t)\right.\right.}\\ &{\left.\left|\mathcal{Q}_{m+1}(0)=0\right.\right.}\\ &{\left.\left|\mathcal{Q}_{m+2}(t)=-\mathcal{L}_{g_{m}}(\mathcal{Q}_{m})\cdot\mathcal{Q}_{m2}(t)+k_{p}\cdot\mathcal{Q}_{m1}(t)\right.\right.}\\ &{\left.\left|\mathcal{Q}_{m+2}(0)=0\right.\right.}\\ &{\left.\left|\mathcal{Q}_{m}\right.\right.\left.-k_{\mathrm{d}},\cdot\mathcal{Q}_{m}(t)+k_{m\mathrm{p}}(\mathcal{Q}_{m})\cdot\mathcal{Q}_{m2}(t)\right.\right.}\\ &{\left.\left|\mathcal{Q}_{m}(0)=0\right.\right.}\\ &{\left.\left.R_{a}(t)=\frac{f}{\mathcal{L}}\cdot k_{m}\cdot\mathcal{Q}_{m}(t)\right.\right.}\\ &{\left.\left|R_{a}(0)=0\right.\right.\right.}\end{array}
$$  

with  

$$
\begin{array}{r l}&{k_{e m p t}(\mathcal{Q}_{s t o})=k_{\operatorname*{min}}+\cfrac{k_{\operatorname*{max}}-k_{\operatorname*{min}}}{2}\,.}\\ &{\left\{\operatorname{tanh}\!\left[\alpha\left(\mathcal{Q}_{s t o}-b\cdot D\right)\right]\!-\!\operatorname{tanh}\!\left[\beta\left(\mathcal{Q}_{s t o}-c\cdot D\right)\right]+2\right\}}\end{array}
$$  

Endogenous glucose production:  

$$
E G P(t)=k_{p1}-k_{p2}\cdot G_{p}(t)-k_{p3}\cdot X^{L}(t)+\xi\cdot X^{H}(t)
$$  

$$
\dot{X}^{L}(t)=-k_{i}\cdot\left[X^{L}(t)-I\left(t\right)\right]\;\;\;X^{L}(0)=I_{b}
$$  

$$
\begin{array}{l}{\dot{I}\,^{\gamma}\!(t)=-k_{i}\cdot\left[I\,^{\gamma}\!(t)-I(t)\right]}\\ {I\,^{\gamma}\!(0)=I_{b}}\end{array}
$$  

$$
\begin{array}{r l}&{\dot{X}^{H}(t)=-k_{H}\cdot X^{H}(t)+k_{H}\cdot\operatorname*{max}\left[\big(H(t)-H_{b}\big),0\right]}\\ &{X^{H}(0)=0}\end{array}
$$  

Glucose utilization:  

$$
U_{i i}(t)=F_{c n s}
$$  

$$
U_{i d}(t)=\frac{\left[V_{m0}+V_{m x}\cdot X(t)\cdot(1+r_{\mathrm{`}}\cdot r i s k)\right]\cdot G_{t}(t)}{K_{m0}+G_{t}(t)}
$$  

$$
\dot{X}(t)=-p_{2U}\cdot X(t)+p_{2U}[I(t)-I_{b}]\,\mathrm{~X(0)=0~}
$$  

$$
r i s k=\left\{\begin{array}{l l}{0}&{\mathrm{if~G\geG_{\mathrm{b}}~}}\\ {10\cdot\left[f(G)\right]^{2}}&{\mathrm{if~\ensuremath{G_{\mathrm{th}}\le G<G_{\mathrm{b}}}~}}\\ {10\cdot\left[f(G_{t h})\right]^{2}}&{\mathrm{if~\ensuremath{G<G_{\mathrm{th}}}~}}\end{array}\right.
$$  

Subcutaneous glucagon kinetics:  

$$
f(G)=\log\left({\frac{G}{G_{b}}}\right)^{r_{2}}
$$  

$$
\begin{array}{r l}&{\bigg[\dot{H}_{s c1}(t)=-\big(k_{h1}+k_{h2}\big)\cdot H_{s c1}(t)}\\ &{\bigg]H_{s c1}(0)=H_{s c1b}}\\ &{\bigg|\dot{H}_{s c2}(t)=k_{h1}\cdot H_{s c1}(t)-k_{h3}\cdot H_{s c2}(t)}\\ &{\bigg|H_{s c2}(0)=H_{s c2b}}\end{array}
$$  

$$
R a_{H}(t)=k_{h3}\cdot H_{s c2}(t)
$$  

Renal excretion:  

$$
E(t)={\left\{\begin{array}{l l}{k_{e1}\cdot[G_{p}(t)-k_{e2}]\ {\mathrm{if}}\ G_{p}(t)>k_{e2}}\\ {0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ {\mathrm{if}\ G_{p}(t)\leq k_{e2}}\end{array}\right.}
$$  

# Abbreviations  

Subcutaneous insulin kinetics:  

$$
\boldsymbol{R}_{a i}(t)=\boldsymbol{k}_{a1}\cdot\boldsymbol{I}_{s c1}(t)+\boldsymbol{k}_{a2}\cdot\boldsymbol{I}_{s c2}(t)
$$  

CF, correction factor; CHO, carbohydrates; CR, insulin to carbs ratio; CSII, continuous subcutaneous insulin infusion; CVGA, control variability grid analysis; D, insulin degradation; E, renal extraction; EGP, endogenous glucose production; FDA, Food and Drug Administration; Ra, glucose rate of appearance; S2008, Type 1 Diabetes Simulator 2008; S2013, Type 1 Diabetes Simulator 2013; SC, rate of insulin appearance from the subcutaneous tissue; T1DM, type 1 diabetes mellitus; U, glucose utilization; UVA, University of Virginia.  

with  

$$
\left\{\begin{array}{l l}{{\dot{I}}_{s c1}({\bf t})=-\big(k_{d}+k_{a1}\big)\cdot I_{s c1}({\bf t})+I I R({\bf t})}\\ {I_{s c1}(0)=I_{s c1s s}}\\ {{\dot{I}}_{s c2}({\bf t})=k_{d}\cdot I_{s c1}({\bf t})-k_{a2}\cdot I_{s c2}({\bf t})}\\ {I_{s c2}(0)=I_{s c2s s}}\end{array}\right.
$$  

# Declaration of Conflicting Interests  

The author(s) declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.  

Subcutaneous glucose kinetics:  

# Funding  

$$
{\dot{\bf G}}_{\mathrm{s}}(\mathrm{t})=-\frac{1}{\mathrm{T}_{\mathrm{s}}}\cdot{\bf G}_{\mathrm{s}}(\mathrm{t})+\frac{1}{\mathrm{T}_{\mathrm{s}}}\cdot{\bf G}(\mathrm{t});\;\;{\bf G}_{\mathrm{s}}(0){=}{\bf G}_{\mathrm{b}}
$$  

The author(s) disclosed receipt of the following financial support for the research, authorship, and/or publication of this article: This study was supported by JDRF grant no. 17-2011-273 and Italian Ministero dell’Università e della Ricerca FIRB 2009.  

Glucagon kinetics and secretion:  

# References  

$$
{\dot{H}}(t)=-n\cdot H(t)+S R_{H}(t)+R a_{H}(t)\qquad H(0)=H_{b}
$$  

with  

$$
S R_{H}(t)=S R_{H}^{s}(t)+S R_{H}^{d}(t)
$$  

$$
\dot{S}R_{H}^{S}(t)=\left\{\begin{array}{l}{-\rho\cdot\left[S R_{H}^{S}(t)-m a x\Big(\sigma_{2}\cdot\left[G t h-G(t)\right]+S R_{H}^{b},0\right)\right]i f\,G(t)\geq G_{b}}\\ {-\rho\cdot\left[S R_{H}^{S}(t)-m a x\Bigg(\frac{\sigma\cdot\left[G t h-G(t)\right]}{I(t)+I}+S R_{H}^{b},0\Bigg)\right]f\,G(t)<G_{b}}\end{array}\right.
$$  

1.	 Cobelli C, Dalla Man C, Sparacino G, Magni L, De Nicolao G, Kovatchev BP. Diabetes: models, signals, and control. IEEE Rev Biomed Eng. 2009;2:54-96.   
2.	 Cobelli C, Renard E, Kovatchev B. Artificial pancreas: past, present, future. Diabetes. 2011;60:2672-2682.   
3.	 Hovorka R, Chassin LJ, Wilinska ME, et al. Closing the loop: the adicol experience. Diab Technol Therap. 2004;3:307-318.   
4.	 Kovatchev BP, Breton M, Dalla Man C, Cobelli C. In silico preclinical trials: a proof of concept in closed-loop control of type 1 diabetes. J Diabetes Sci Technol. 2009;3(1):44-55.   
5.	 Visentin  R,  Dalla  Man  C,  Kovatchev  BP,  Cobelli  C. Incorporating nonlinear response to hypoglycemia into the type 1 diabetes simulator. Paper presented at: 11th Diabetes Technology Meeting; October 27-29, 2011; San Francisco, CA.   
6.	 Chan A, Heinemann L, Anderson S, Breton MD, Kovatchev BP. Nonlinear metabolic effect of insulin across the blood glucose range in patients with type 1 diabetes mellitus. J Diabetes Sci Technol. 2010;4:873-881.   
7.	 Micheletto F, Dalla Man C, Breton MD, Kovatchev BP, Cobelli C. A counter-regulation model in type 1 diabetes. Paper  

$$
\mathrm{SR}_{\mathrm{H}}^{\mathrm{~d~}}(t)=\boldsymbol{\delta}\cdot\operatorname*{max}\left(-\frac{d G(t)}{d t},0\right)
$$  

presented at: 11th Diabetes Technology Meeting; October 27- 29, 2011; San Francisco, CA.   
8.	 Dobbins RL, Davis SN, Neal DW, Cobelli C, Jaspan J, Cherrington AD. Compartmental modeling of glucagon kinetics in the conscious dog. Metabolism. 1995;44:452-459.   
9.	 Kovatchev BP, Farhy LS, Cox DJ, et al. Modeling insulinglucose  dynamics  during  insulin  induced  hypoglycemia. Evaluation  of  glucose  counterregulation.  J  Theor  Med. 1999;1:313-323.   
10.	 Kovatchev BP, Straume M, Farhy LS, Cox DJ. Dynamic network model of glucose counterregulation in subjects with insulin-requiring diabetes. In: Johnson M, Brand L, eds. Methods in Enzymology, 321: Numerical Computer Methods, Part C. New York, NY: Academic Press; 2000:396-410.   
11.	 Kovatchev BP, Straume M, Cox DJ, Farhy LS. Risk analysis of blood glucose data: a quantitative approach to optimizing the control of insulin dependent diabetes. J Theor Med. 1999;3:1- 10.   
12.	 Bergman RN, Ider YZ, Bowden CR, Cobelli C. Quantitative estimation of insulin sensitivity. Am J Physiol. 1979;236(6):E66 7-E677.   
13.	 El-Khatib FH, Russell SJ, Nathan DM, Sutherlin RG, Damiano ER. A bihormonal closed-loop artificial pancreas for type 1 diabetes. Sci Transl Med. 2010;2(27):27ra27.   
14.	 Lv D, Breton MD, Farhy LS. Pharmacokinetics modeling of exogenous glucagon in type 1 diabetes mellitus patients. Diabetes Technol Ther. 2013;15(11):935-941.   
15.	 Lorenzi M, Bohannon N, Tsalikian E, et al. Duration of type 1 diabetes affects glucagon and glucose responses to insulininduced hypoglycemia. West J Med. 1984;141:467-471.   
16.	 Siafarikas A, Johnston RJ, Bulsara MK, et al. Early loss of glucagon response to hypoglycemia in adolescents with type 1 diabetes. Diabetes Care. 2012;35(8):1757-1762.   
17.	 Sjöberg A, Ahren B, Bolinder J. Residual insulin secretion is not coupled to a maintained glucagon response to hypoglycaemia in long-term type 1 diabetes. J Intern Med. 2002;252:342-351.   
18.	 Bruno G, Runzo C, Cavallo-Perin P, et al. Incidence of type 1 and type 2 diabetes in adults aged 30-49 years. Diabetes Care. 2005;28:2613-2619.   
19.	 Felner EI, Klitz W, Ham M, et al. Genetic interaction among three genomic regions creates distinct contributions to earlyand late-onset type 1 diabetes mellitus. Pediatr Diabetes. 2005;6:213-220.   
20.	 Laakso M, Pyörälä K. Age of onset and type of diabetes. Diabetes Care. 1985;8(2):114-117.   
21.	 Karjalainen J, Salmela P, Ilonen J, et al. A comparison of childhood and adult type 1 diabetes mellitus. N Engl J Med. 1989;320(14):881-886.   
22.	 Tenconi MT, Devoti G, Albani I, et al. IDDM in the province of Pavia, Italy, from a population-based registry. Diabetes Care. 1995;18(7):1017-1019.   
23.	 Hinshaw L, Dalla Man C, Nandy DK, et al. Diurnal pattern of insulin action in type 1 diabetes: implications for a closed-loop system. Diabetes. 2013;62(7):2223-2229.   
24.	 Warshaw  H,  Bolderman  KM.  Practical  Carbohydrate Counting: A How-to-Teach Guide for Health Professionals. Alexandria, VA: American Diabetes Association; 2001.   
25.	 Walsh J, Roberts R. Setting and testing your carb boluses. In: Torrey Pines Press, ed. Pumping Insulin. Everything You Need for Success with an Insulin Pump. 3rd ed. San Diego, CA: Torrey Pines Press; 2000:105-113.   
26.	 Bolderman  KM.  Prepping  pump  patients.  In:  American Diabetes Association, ed. Putting Your Patients on the Pump. Alexandria, VA: American Diabetes Association; 2002:19-38.   
27.	 Davidson PC, Hebblewhite HR, Bode BW, et al. Statisticallybased CSII parameters: correction factor, CF (1700 rule), carbohydrate-insulin ratio, CIR (2,8 rule), and basal-to-total ratio (abstract). Diabetes Technol Ther. 2003;5(2):237.   
28.	 Magni L, Raimondo DM, Dalla Man C, et al. Evaluating the efficiancy of closed-loop glucose regulation via controlvariability grid analysis (CVGA). J Diabetes Sci Technol. 2008;2:630-635.  