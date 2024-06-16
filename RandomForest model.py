from tkinter import *
import numpy as np
import pandas as pd
from sklearn.tree import export_text, plot_tree
import graphviz
import matplotlib.pyplot as plt

#List of the symptoms is listed here in list l1.

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

#List of Diseases is listed in list disease.

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]

for i in range(0,len(l1)):
    l2.append(0)

df=pd.read_csv("/Users/prakashgowdatr/Downloads/Prototype.csv")

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

#check the df 
#print(df.head())

X= df[l1]

#print(X)

y = df[["prognosis"]]
np.ravel(y)

#print(y)

#Read a csv named Testing.csv

tr=pd.read_csv("/Users/prakashgowdatr/Downloads/Prototype-1.csv")

#Use replace method in pandas.

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]

#print(y_test)

np.ravel(y_test)


def randomforest():
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy 
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    # Plot feature importance
    feature_importance = clf4.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), X.columns[sorted_idx])
    plt.xlabel('Importance')
    plt.title('Feature importances')
    plt.draw()
    plt.show() 
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    selected_symptoms = [symptom for symptom in psymptoms if symptom != "Select Symptom"]

    # Check if at least one symptom is selected
    if not selected_symptoms:
        t2.delete("1.0", END)
        t2.insert(END, "Please select at least one symptom.")
        return

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END,"Predicted disease: "+disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")





# GUI stuff..............................................................................
        
import customtkinter as ctk
from tkinter import ttk

root = ctk.CTk()
root.title("DPred_GPT")
root.geometry("1100x600")  # Width x Height

title = ctk.CTkLabel(root, text="Medical Diagnosis System using RandomForest \n Team 12", bg_color='light grey', fg_color='grey18')
title.configure(font=("Times New Roman", 30))  # Set the font size and style
title.grid(row=0, column=1, columnspan=2, pady=15)  # Place it at the top of the window


# Create style object
style = ttk.Style()


# Define variables
Symptom1 = StringVar()


Symptom2 = StringVar()


Symptom3 = StringVar()


Symptom4 = StringVar()


Symptom5 = StringVar()


Name = StringVar()
OPTIONS=sorted(l1)

# Define labels, entries, and buttons
NameLb = ctk.CTkLabel(root, text="  Name of the Patient : ",font=('Helvetica', 20), bg_color='light grey', fg_color='grey18')
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = ctk.CTkLabel(root, text="  Symptom 1  ",font=('Helvetica', 20,), bg_color='light grey', fg_color='gray20')
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = ctk.CTkLabel(root, text="  Symptom 2  ",font=('Helvetica', 20,), bg_color='light grey', fg_color='gray20')
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = ctk.CTkLabel(root, text="  Symptom 3  ",font=('Helvetica', 20,), bg_color='light grey', fg_color='gray20')
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = ctk.CTkLabel(root, text="  Symptom 4  ",font=('Helvetica', 20,), bg_color='light grey', fg_color='gray20')
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = ctk.CTkLabel(root, text="  Symptom 5  ",font=('Helvetica', 20,), bg_color='light grey', fg_color='gray20')
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


rnf = ctk.CTkButton(root, text="RandomForest prediction", command=randomforest,font=('Helvetica', 20), bg_color='blue', fg_color='dodger blue3')
rnf.grid(row=9, column=3,padx=10)


NameEn = ttk.Entry(root, textvariable=Name,font=('Helvetica', 20))
NameEn.grid(row=6, column=1)


S1 = ttk.OptionMenu(root, Symptom1, *OPTIONS)
S1.grid(row=7, column=1)

S2 = ttk.OptionMenu(root, Symptom2, *OPTIONS)
S2.grid(row=8, column=1)

S3 = ttk.OptionMenu(root, Symptom3, *OPTIONS)
S3.grid(row=9, column=1)

S4 = ttk.OptionMenu(root, Symptom4, *OPTIONS)
S4.grid(row=10, column=1)

S5 = ttk.OptionMenu(root, Symptom5, *OPTIONS)
S5.grid(row=11, column=1)

Symptom1.set("Select Symptom")
Symptom2.set("Select Symptom")
Symptom3.set("Select Symptom")
Symptom4.set("Select Symptom")
Symptom5.set("Select Symptom")


t2 = Text(root, height=1, width=40,bg="dodger blue3",fg="white smoke")
t2.config(font=("Verdana",20))
t2.grid(row=17, column=1 , padx=10)

authors = ctk.CTkLabel(root, text="AUTHORS \n Prakash_CH22B085", bg_color='light grey', fg_color='grey18')
authors.configure(font=("Helvetica", 10))  # Set the font size and style
authors.grid(row=19, column=3, columnspan=2, pady=15)  # Place it at the bottom of the window

root.mainloop()

