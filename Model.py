import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def model(x):

    #Read the data
    df = pd.read_csv('Data_1dpb_2dpt_6purp.csv')
    INCOME_SCALED = df.FAM_INC_imputed/1000
    df['student'] = df.preschool_student+df.primaryschool_student+df.highschool_student
    df['student']= df['student'].astype(int)

    # BINARIES FOR TOUR NUMBER, PURPOSE, AND COMBINATION INPUTS

    # binaries for number of tours in each option
    onetour = [0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    twotour = [0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    threetour = [0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0]
    fourtour = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0]
    #FiveTours = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0]
    #SixTours = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]

    # binaries for existance of that specific purpose of tour
    WorkT = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    EduT = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    PersonalT = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    ShopT = [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
    LeisureT = [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
    EscortT = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]

    # binary for tour purpose combination
    workedu_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    workpersonal_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    workshop_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
    workleisure_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
    workescort_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    edupersonal_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
    edushop_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]
    eduleisure_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1]
    eduescort_tt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    personalshop_tt = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
    personalleisure_tt = [0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1]
    personalescort_tt = [0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,1]
    shopleisure_tt = [0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1]
    shopescort_tt = [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,1]
    leisureescort_tt = [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]

    beta_LIC_edu = 	0.00766
    beta_LIC_escort	= 1.13
    beta_LIC_personal= 0.225	
    beta_LIC_rec= 	0.224	
    beta_LIC_shop= 	0.114	
    beta_OnLeave_edu= 	-1.86
    beta_OnLeave_escort	= 2.44	
    beta_OnLeave_leisure= 	1.61	
    beta_OnLeave_personal= 	2.05	
    beta_OnLeave_shop= 	1.71	
    beta_OtherStudent_edu	= 7.51	
    beta_OtherStudent_escort= 	1.34	
    beta_OtherStudent_leisure= 	1.27	
    beta_OtherStudent_personal= 	1.32	
    beta_OtherStudent_shop= 	1.67	
    beta_TRANS_edu	= -0.224	
    beta_TRANS_escort= 	-0.0304
    beta_TRANS_personal	= 0.0852
    beta_TRANS_rec= 	0.0856
    beta_TRANS_shop	= -0.014	
    beta_Trainee_edu= 	5.65	
    beta_Trainee_escort= 	1.09	
    beta_Trainee_leisure= 	0.257	
    beta_Trainee_personal= 	0.758	
    beta_Trainee_shop= 	0.871
    beta_age2025_edu= 	-0.204	
    beta_age2025_escort	= -1.78	
    beta_age2025_leisure= 	0.13	
    beta_age2025_personal= 	-0.146	
    beta_age2025_shop= 	-0.189	
    beta_age2635_edu= 	-0.278
    beta_age2635_escort= 	-0.55	
    beta_age2635_leisure= -0.26	
    beta_age2635_personal= 	-0.352	
    beta_age2635_shop	= -0.216	
    beta_age5165_edu= 	-0.489	
    beta_age5165_escort= 	-1.34	
    beta_age5165_leisure= 	0.0155
    beta_age5165_personal= 	0.119
    beta_age5165_shop= 	0.043
    beta_age65_edu	= -0.113
    beta_age65_escort= 	-1.85
    beta_age65_personal = 	0.37
    beta_age65_shop	 = 0.0558
    beta_ageUpto19_edu	= -0.00682
    beta_ageUpto19_escort	= -2.02
    beta_ageUpto19_leisure	= 0.0389
    beta_ageUpto19_personal	= -0.231
    beta_ageUpto19_shop	= -0.832
    beta_disabled_edu	= -3.41
    beta_disabled_escort	= 0.832
    beta_disabled_leisure	= 1.57
    beta_disabled_personal	= 2.22
    beta_disabled_shop	= 2.27
    beta_eduescort_tt	= -0.872
    beta_eduleisure_tt	= -0.38
    beta_edupersonal_tt	= -0.91
    beta_edushop_tt	= -0.976
    beta_fam_income_edu	= 0.0538
    beta_fam_income_escort	= 0.0531
    beta_fam_income_leisure	= 0.0422
    beta_fam_income_personal= 	0.0229
    beta_fam_income_shop	= 0.0329
    beta_female_edu	= 0.0208
    beta_female_escort	= 0.253
    beta_female_leisure	= 0.0706
    beta_female_personal	= -0.0624
    beta_female_shop= 	0.185
    beta_fourtours = 	-4.52
    beta_homemaker_edu	= -2.38
    beta_homemaker_escort= 	1.65
    beta_homemaker_leisure= 1.93
    beta_homemaker_personal= 1.1
    beta_homemaker_shop	= 1.92
    beta_leisureescort_tt= 	0.269
    beta_parttime_edu= 	1.44
    beta_parttime_escort= 	0.497
    beta_parttime_leisure= 	0.452
    beta_parttime_personal= 	0.764
    beta_parttime_shop	= 0.548
    beta_personalescort_tt	= -0.196
    beta_personalleisure_tt	= 0.0174
    beta_personalshop_tt	= -0.457
    beta_retired_edu	= -6.27
    beta_retired_escort	= 0.964	
    beta_retired_leisure= 	1.85
    beta_retired_personal= 	1.73
    beta_retired_shop	= 2.2
    beta_shopescort_tt	= 0.54
    beta_shopleisure_tt	= -0.015
    beta_student_edu	= 8.11
    beta_student_escort	= 1.33
    beta_student_leisure= 	1.85
    beta_student_personal= 	1.25
    beta_student_shop	= 1.73	
    beta_threetour	= -2.64
    beta_tour_edu	= -5.35
    beta_tour_escort	= -3.48
    beta_tour_leisure= 	-1.34
    beta_tour_personal= -3
    beta_tour_shop	= -1.63
    beta_twotours	= -0.935
    beta_unemployed_edu	= -2.85
    beta_unemployed_escort = 	1.48
    beta_unemployed_leisure	 = 1.66
    beta_unemployed_personal= 	2.24
    beta_unemployed_shop	= 2.24
    beta_universitystudent_edu	= 6.47
    beta_universitystudent_escort	= 1.1
    beta_universitystudent_leisure	= 0.969	
    beta_universitystudent_personal	= 1.25
    beta_universitystudent_shop	= 1.54
    beta_workescort_tt	= 0.705
    beta_workleisure_tt	= 0.165	
    beta_workpersonal_tt= 	-0.913
    beta_workshop_tt	= -0.298

    beta_LIC_travel=0.78
    beta_OnLeave_travel=-1.24
    beta_OtherStudent_travel=-0.0601
    beta_TRANS_travel= 0.391
    beta_age2025_travel= -0.274
    beta_age2635_travel=0.0252
    beta_age5165_travel	=-0.232
    beta_age65_travel=	-0.571
    beta_ageUpto19_travel =	-0.186
    beta_disabled_travel= -1.5
    beta_fam_income_travel	= 0.000143
    beta_female_travel	 = 0.099
    beta_homemaker_travel= -1.73
    beta_parttime_travel= -0.155
    beta_retired_travel	= -0.988
    beta_student_travel	= 0.61
    beta_unemployed_travel	= -1.28
    beta_universitystudent_travel= -0.436
    cons_travel	= 1.54



    beta_female_travel = 0.099
    beta_TRANS_travel= 0.391
    beta_student_travel	= 0.61
    
    #Change the beta values
    beta_female_travel = beta_female_travel * x[0]
    beta_student_travel = beta_student_travel * x[1]
    beta_TRANS_travel = beta_TRANS_travel * x[2]

    # Utility equations
    #V33 is the base: work=1, numberoftours=1 (V33=0)
    V= []
    for i in range(1,63): #generate utility functions for all activity patterns (although not all will be used int the model)
        V.append( beta_tour_edu * (EduT[i]) + beta_tour_personal * (PersonalT[i]) +\
            beta_tour_leisure * (LeisureT[i]) + beta_tour_shop * (ShopT[i]) + beta_tour_escort * (EscortT[i])+\
            beta_twotours * (twotour[i]) +\
            beta_fourtours * (fourtour[i]) +\
            beta_workpersonal_tt * (workpersonal_tt[i]) + beta_workleisure_tt * (workleisure_tt[i]) +\
            beta_workshop_tt * (workshop_tt[i]) + beta_workescort_tt * (workescort_tt[i]) +\
            beta_edupersonal_tt * (edupersonal_tt[i]) + beta_edushop_tt * (edushop_tt[i]) + beta_eduleisure_tt * (eduleisure_tt[i]) +\
            beta_eduescort_tt * (eduescort_tt[i]) +\
            beta_personalshop_tt * (personalshop_tt[i]) + beta_personalleisure_tt * (personalleisure_tt[i]) + beta_personalescort_tt * (personalescort_tt[i]) +\
            beta_shopleisure_tt * (shopleisure_tt[i]) + beta_shopescort_tt * (shopescort_tt[i]) +\
            beta_leisureescort_tt * (leisureescort_tt[i]) +\
            beta_parttime_edu * (EduT[i] * df.parttime) + beta_parttime_personal * (PersonalT[i] * df.parttime) +\
            beta_parttime_leisure * (LeisureT[i] * df.parttime) + beta_parttime_shop * (ShopT[i] * df.parttime) + beta_parttime_escort * (EscortT[i] * df.parttime) +\
            beta_retired_edu * (EduT[i] * df.retired) + beta_retired_personal * (PersonalT[i] * df.retired) +\
            beta_retired_leisure * (LeisureT[i] * df.retired) + beta_retired_shop * (ShopT[i] * df.retired) + beta_retired_escort * (EscortT[i] * df.retired)  +\
            beta_disabled_edu * (EduT[i] * df.disabled) + beta_disabled_personal * (PersonalT[i] * df.disabled) +\
            beta_disabled_leisure * (LeisureT[i] * df.disabled) + beta_disabled_shop * (ShopT[i] * df.disabled) + beta_disabled_escort * (EscortT[i] * df.disabled)  +\
            beta_homemaker_edu * (EduT[i] * df.homemaker) + beta_homemaker_personal * (PersonalT[i] * df.homemaker) +\
            beta_homemaker_leisure * (LeisureT[i] * df.homemaker) + beta_homemaker_shop * (ShopT[i] * df.homemaker) + beta_homemaker_escort * (EscortT[i] * df.homemaker)  +\
            beta_OnLeave_edu * (EduT[i] * df.onLeave) + beta_OnLeave_personal * (PersonalT[i] * df.onLeave) +\
            beta_OnLeave_leisure * (LeisureT[i] * df.onLeave) + beta_OnLeave_shop * (ShopT[i] * df.onLeave) + beta_OnLeave_escort * (EscortT[i] * df.onLeave)  +\
            beta_unemployed_edu * (EduT[i] * df.unemployed) + beta_unemployed_personal * (PersonalT[i] * df.unemployed) +\
            beta_unemployed_leisure * (LeisureT[i] * df.unemployed) + beta_unemployed_shop * (ShopT[i] * df.unemployed) + beta_unemployed_escort * (EscortT[i] * df.unemployed)  +\
            beta_universitystudent_edu * (EduT[i] * df.universityStudent) +\
            beta_universitystudent_personal * (PersonalT[i] * df.universityStudent) + beta_universitystudent_leisure * (LeisureT[i] * df.universityStudent) +\
            beta_universitystudent_shop * (ShopT[i] * df.universityStudent) + beta_universitystudent_escort * (EscortT[i] * df.universityStudent)  +\
            beta_student_edu * (EduT[i] * df.student) + beta_student_personal * (PersonalT[i] * df.student) +\
            beta_student_leisure * (LeisureT[i] * df.student) + beta_student_shop * (ShopT[i] * df.student) + beta_student_escort * (EscortT[i] * df.student)  +\
            beta_OtherStudent_edu * (EduT[i] * df.other_student) + beta_OtherStudent_personal * (PersonalT[i] * df.other_student) +\
            beta_OtherStudent_leisure * (LeisureT[i] * df.other_student) + beta_OtherStudent_shop * (ShopT[i] * df.other_student) + beta_OtherStudent_escort * (EscortT[i] * df.other_student)  +\
            beta_Trainee_edu * (EduT[i] * df.trainee) + beta_Trainee_personal * (PersonalT[i] * df.trainee) +\
            beta_Trainee_leisure * (LeisureT[i] * df.trainee) + beta_Trainee_shop * (ShopT[i] * df.trainee) + beta_Trainee_escort * (EscortT[i] * df.trainee)  +\
            beta_ageUpto19_edu * (EduT[i] * df.ageUpto19) + beta_ageUpto19_personal * (PersonalT[i] * df.ageUpto19) +\
            beta_ageUpto19_leisure * (LeisureT[i] * df.ageUpto19) + beta_ageUpto19_shop * (ShopT[i] * df.ageUpto19) + beta_ageUpto19_escort * (EscortT[i] * df.ageUpto19) +\
            beta_age2025_edu * (EduT[i] * df.age2025) + beta_age2025_personal * (PersonalT[i] * df.age2025) +\
            beta_age2025_leisure * (LeisureT[i] * df.age2025) + beta_age2025_shop * (ShopT[i] * df.age2025) + beta_age2025_escort * (EscortT[i] * df.age2025)  +\
            beta_age2635_edu * (EduT[i] * df.age2635) + beta_age2635_personal * (PersonalT[i] * df.age2635) +\
            beta_age2635_leisure * (LeisureT[i] * df.age2635) + beta_age2635_shop * (ShopT[i] * df.age2635) + beta_age2635_escort * (EscortT[i] * df.age2635)  +\
            beta_age5165_edu * (EduT[i] * df.age5165) + beta_age5165_personal * (PersonalT[i] * df.age5165) +\
            beta_age5165_leisure * (LeisureT[i] * df.age5165) + beta_age5165_shop * (ShopT[i] * df.age5165) + beta_age5165_escort * (EscortT[i] * df.age5165)  +\
            beta_age65_edu * (EduT[i] * df.ageMorethan65) + beta_age65_personal * (PersonalT[i] * df.ageMorethan65) +\
            beta_age65_shop * (ShopT[i] * df.ageMorethan65) + beta_age65_escort * (EscortT[i] * df.ageMorethan65)  +\
            beta_female_edu * (EduT[i] * df.female) + beta_female_personal * (PersonalT[i] * df.female) +\
            beta_female_leisure * (LeisureT[i] * df.female) + beta_female_shop * (ShopT[i] * df.female) + beta_female_escort * (EscortT[i] * df.female)  +\
            beta_fam_income_edu * (EduT[i] * INCOME_SCALED) + beta_fam_income_personal * (PersonalT[i] * INCOME_SCALED) +\
            beta_fam_income_leisure * (LeisureT[i] * INCOME_SCALED) + beta_fam_income_shop * (ShopT[i] * INCOME_SCALED) + beta_fam_income_escort * (EscortT[i] * INCOME_SCALED)  +\
            beta_LIC_edu * (EduT[i] * df.DRVLC) + beta_LIC_personal * (PersonalT[i] * df.DRVLC) + beta_LIC_rec * (LeisureT[i] * df.DRVLC) +\
            beta_LIC_shop * (ShopT[i] * df.DRVLC) + beta_LIC_escort * (EscortT[i] * df.DRVLC)  +\
            beta_TRANS_edu * (EduT[i] * df.PTPASS) + beta_TRANS_personal * (PersonalT[i] * df.PTPASS) +\
            beta_TRANS_rec * (LeisureT[i] * df.PTPASS) + beta_TRANS_shop * (ShopT[i] * df.PTPASS) + beta_TRANS_escort * (EscortT[i] * df.PTPASS))
    V = np.array(V)
    Ve= math.e**(V)
    dem = sum(Ve)
    probs=np.array([math.e**(V[i,:])/dem for i in range(62)])
    V1 = np.zeros(len(df))
    V2 = cons_travel +\
    beta_parttime_travel * df.parttime +\
    beta_retired_travel * df.retired +\
    beta_disabled_travel * df.disabled +\
    beta_homemaker_travel * df.homemaker +\
    beta_OnLeave_travel * df.onLeave +\
    beta_unemployed_travel * df.unemployed +\
    beta_universitystudent_travel * df.universityStudent +\
    beta_student_travel * df.student +\
    beta_OtherStudent_travel * df.other_student +\
    beta_ageUpto19_travel * df.ageUpto19 +\
    beta_age2025_travel * df.age2025 +\
    beta_age2635_travel * df.age2635 +\
    beta_age5165_travel * df.age5165 +\
    beta_age65_travel * df.ageMorethan65 +\
    beta_female_travel * df.female +\
    beta_fam_income_travel * df.FAM_INC_imputed +\
    beta_LIC_travel * df.DRVLC +\
    beta_TRANS_travel * df.PTPASS
    dem= (math.e**(V1)+math.e**(V2))
    probs1=np.array([math.e**(V1)/dem,math.e**(V2)/dem])
    p_stay= probs1[0].reshape((1, len(df)))
    new_probs= probs1[1]*probs
    new_probs= np.concatenate((p_stay, new_probs), axis= 0)
    choices = [np.random.choice(np.arange(0, 63), p = new_probs[:,i]) for i in range(len(df))]
    code= [1 if LeisureT[choices[i]] ==1 else 0 for i in range(len(df))]
    output = sum(code) #the return is the number of trips that include leisure

    return output

