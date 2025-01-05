About Dataset
Overview:
The AIDS Clinical Trials Group Study 175 Dataset, initially published in 1996, is a comprehensive collection of healthcare statistics and categorical information about patients diagnosed with AIDS. This dataset was created with the primary purpose of examining the performance of two different types of AIDS treatments: zidovudine (AZT) versus didanosine (ddI), AZT plus ddI, and AZT plus zalcitabine (ddC). The prediction task associated with this dataset involves determining whether each patient died within a specified time window.

Dataset Details:

Number of rows: 2139
Number of columns: 24
Purpose of Dataset Creation:
The dataset was created to evaluate the efficacy and safety of various AIDS treatments, specifically comparing the performance of AZT, ddI, and ddC in preventing disease progression in HIV-infected patients with CD4 counts ranging from 200 to 500 cells/mm3. This intervention trial aimed to contribute insights into the effectiveness of monotherapy versus combination therapy with nucleoside analogs.

Funding Sources:
The creation of this dataset was funded by:

AIDS Clinical Trials Group of the National Institute of Allergy and Infectious Diseases
General Research Center units funded by the National Center for Research Resources
Instance Representation:
Each instance in the dataset represents a health record of a patient diagnosed with AIDS in the United States. These records encompass crucial categorical information and healthcare statistics related to the patient's condition.

Study Design:

Study Type: Interventional (Clinical Trial)
Enrollment: 2100 participants
Masking: Double-Blind
Primary Purpose: Treatment
Official Title: A Randomized, Double-Blind Phase II/III Trial of Monotherapy vs. Combination Therapy With Nucleoside Analogs in HIV-Infected Persons With CD4 Cells of 200-500/mm3
Study Completion Date: November 1995
Study Objectives:
To determine the effectiveness and safety of different AIDS treatments, including AZT, ddI, and ddC, in preventing disease progression among HIV-infected patients with specific CD4 cell counts.

Additional Information:
The dataset provides valuable insights into the HIV-related clinical trials conducted by the AIDS Clinical Trials Group, contributing to the understanding of treatment outcomes and informing future research in the field.

Attributes Description:

Patient Information:
Censoring Indicator (label):Binary indicator (1 = failure, 0 = censoring) denoting patient status.
Temporal Information:
Time to Event (time): Integer representing time to failure or censoring.
Treatment Features:
Treatment Indicator (trt): Categorical feature indicating the type of treatment received (0 = ZDV only, 1 = ZDV + ddI, 2 = ZDV + Zal, 3 = ddI only).
Baseline Health Metrics:
Age (age): Patient's age in years at baseline.
Weight (wtkg): Continuous feature representing weight in kilograms at baseline.
Hemophilia (hemo): Binary indicator of hemophilia status (0 = no, 1 = yes).
Sexual Orientation (homo): Binary indicator of homosexual activity (0 = no, 1 = yes).
IV Drug Use History (drugs): Binary indicator of history of IV drug use (0 = no, 1 = yes).
Karnofsky Score (karnof): Integer on a scale of 0-100 indicating the patient's functional status.
Antiretroviral Therapy History:
Non-ZDV Antiretroviral Therapy Pre-175 (oprior): Binary indicator of non-ZDV antiretroviral therapy pre-Study 175 (0 = no, 1 = yes).
ZDV in the 30 Days Prior to 175 (z30): Binary indicator of ZDV use in the 30 days prior to Study 175 (0 = no, 1 = yes).
ZDV Prior to 175 (zprior): Binary indicator of ZDV use prior to Study 175 (0 = no, 1 = yes).
Days Pre-175 Anti-Retroviral Therapy (preanti): Integer representing the number of days of pre-Study 175 anti-retroviral therapy.
Demographic Information:
Race (race): Integer denoting race (0 = White, 1 = non-white).
Gender (gender): Binary indicator of gender (0 = Female, 1 = Male).
Treatment History:
Antiretroviral History (str2): Binary indicator of antiretroviral history (0 = naive, 1 = experienced).
Antiretroviral History Stratification (strat): Integer representing antiretroviral history stratification.
Symptomatic Information:
Symptomatic Indicator (symptom): Binary indicator of symptomatic status (0 = asymptomatic, 1 = symptomatic).
Additional Treatment Attributes:
Treatment Indicator (treat): Binary indicator of treatment (0 = ZDV only, 1 = others).
Off-Treatment Indicator (offtrt): Binary indicator of being off-treatment before 96+/-5 weeks (0 = no, 1 = yes).
Immunological Metrics:
CD4 Counts (cd40, cd420): Integer values representing CD4 counts at baseline and 20+/-5 weeks.
CD8 Counts (cd80, cd820): Integer values representing CD8 counts at baseline and 20+/-5 weeks.
Original Dataset Website:
https://classic.clinicaltrials.gov/ct2/show/NCT00000625