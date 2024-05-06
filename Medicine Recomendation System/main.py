from flask import Flask,request,render_template #The Flask class is the core component of a Flask web application. It allows you to create a Flask instance (your web app) that can handle routes, views, and other functionalities.
import numpy as np  #It provides powerful tools for working with numerical data, especially arrays and matrices.
import pandas as pd  #It provides data structures (such as DataFrames and Series) that allow you to work with structured data efficiently.
import pickle   #Pickle is a Python module that allows you to serialize (convert into a byte stream) and deserialize (convert back from a byte stream)
                 #Scikit-learn is built on popular Python libraries such as NumPy, SciPy, and Matplotlib, making it a powerful tool for both beginners and experienced machine learning practitioners
#load database===========================================
sys_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd .read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description=pd.read_csv("datasets/description.csv")
medications=pd.read_csv("datasets/medications.csv")
diets=pd.read_csv("datasets/diets.csv")

#load models=============================================
svc = pickle.load(open("models/svc.pkl",'rb'))    #SVC stands for Support Vector Classifier. It is a machine learning algorithm based on the concept of support vector machines (SVMs)

app=Flask(__name__)                               #The @app.route('/') decorator defines a route for the root URL (“/”).


# ===================================================================
# custome and helping functions
# =================================helper functions==================
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])                  # description This list comprehension splits the description into individual words (tokens).

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis]['workout']
    return desc, pre, med, die, wrkout


# model prediction function
symptoms_dict = {'itching': 0,
                 'skin_rash': 1,
                 'nodal_skin_eruptions': 2,
                 'continuous_sneezing': 3,
                 'shivering': 4,
                 'chills': 5,
                 'joint_pain': 6,
                 'stomach_pain': 7,
                 'acidity': 8,
                 'ulcers_on_tongue': 9,
                 'muscle_wasting': 10,
                 'vomiting': 11,
                 'burning_micturition': 12,
                 'spotting_urination': 13,
                 'fatigue': 14,
                 'weight_gain': 15,
                 'anxiety': 16,
                 'cold_hands_and_feets': 17,
                 'mood_swings': 18,
                 'weight_loss': 19,
                 'restlessness': 20,
                 'lethargy': 21,
                 'patches_in_throat': 22,
                 'irregular_sugar_level': 23,
                 'cough': 24,
                 'high_fever': 25,
                 'sunken_eyes': 26,
                 'breathlessness': 27,
                 'sweating': 28,
                 'dehydration': 29,
                 'indigestion': 30,
                 'headache': 31,
                 'yellowish_skin': 32,
                 'dark_urine': 33,
                 'nausea': 34,
                 'loss_of_appetite': 35,
                 'pain_behind_the_eyes': 36,
                 'back_pain': 37,
                 'constipation': 38,
                 'abdominal_pain': 39,
                 'diarrhoea': 40,
                 'mild_fever': 41,
                 'yellow_urine': 42,
                 'yellowing_of_eyes': 43,
                 'acute_liver_failure': 44,
                 'fluid_overload': 45,
                 'swelling_of_stomach': 46,
                 'swelled_lymph_nodes': 47,
                 'malaise': 48,
                 'blurred_and_distorted_vision': 49,
                 'phlegm': 50,
                 'throat_irritation': 51,
                 'redness_of_eyes': 52,
                 'sinus_pressure': 53,
                 'runny_nose': 54,
                 'congestion': 55,
                 'chest_pain': 56,
                 'weakness_in_limbs': 57,
                 'fast_heart_rate': 58,
                 'pain_during_bowel_movements': 59,
                 'pain_in_anal_region': 60,
                 'bloody_stool': 61,
                 'irritation_in_anus': 62,
                 'neck_pain': 63,
                 'dizziness': 64,
                 'cramps': 65,
                 'bruising': 66,
                 'obesity': 67,
                 'swollen_legs': 68,
                 'swollen_blood_vessels': 69,
                 'puffy_face_and_eyes': 70,
                 'enlarged_thyroid': 71,
                 'brittle_nails': 72,
                 'swollen_extremeties': 73,
                 'excessive_hunger': 74,
                 'extra_marital_contacts': 75,
                 'drying_and_tingling_lips': 76,
                 'slurred_speech': 77,
                 'knee_pain': 78,
                 'hip_joint_pain': 79,
                 'muscle_weakness': 80,
                 'stiff_neck': 81,
                 'swelling_joints': 82,
                 'movement_stiffness': 83,
                 'spinning_movements': 84,
                 'loss_of_balance': 85,
                 'unsteadiness': 86,
                 'weakness_of_one_body_side': 87,
                 'loss_of_smell': 88,
                 'bladder_discomfort': 89,
                 'foul_smell_ofurine': 90,
                 'continuous_feel_of_urine': 91,
                 'passage_of_gases': 92,
                 'internal_itching': 93,
                 'toxic_look_(typhos)': 94,
                 'depression': 95,
                 'irritability': 96,
                 'muscle_pain': 97,
                 'altered_sensorium': 98,
                 'red_spots_over_body': 99,
                 'belly_pain': 100,
                 'abnormal_menstruation': 101,
                 'dischromic_patches': 102,
                 'watering_from_eyes': 103,
                 'increased_appetite': 104,
                 'polyuria': 105,
                 'family_history': 106,
                 'mucoid_sputum': 107,
                 'rusty_sputum': 108,
                 'lack_of_concentration': 109,
                 'visual_disturbances': 110,
                 'receiving_blood_transfusion': 111,
                 'receiving_unsterile_injections': 112,
                 'coma': 113,
                 'stomach_bleeding': 114,
                 'distention_of_abdomen': 115,
                 'history_of_alcohol_consumption': 116,
                 'fluid_overload': 117,
                 'blood_in_sputum': 118,
                 'prominent_veins_on_calf': 119,
                 'palpitations': 120,
                 'painful_walking': 121,
                 'pus_filled_pimples': 122,
                 'blackheads': 123,
                 'scurring': 124,
                 'skin_peeling': 125,
                 'silver_like_dusting': 126,
                 'small_dents_in_nails': 127,
                 'inflammatory_nails': 128,
                 'blister': 129,
                 'red_sore_around_nose': 130,
                 'yellow_crust_ooze': 131,
                 'prognosis': 132,
                 }

diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
                 33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes', 17: 'Gastroenterits', 6: 'Bronchial Asthma',
                 23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
                 28: 'Jaundice',
                 29: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A',
                 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 25: 'Hypoglycemia',
                 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo', 2: 'Acne',
                 38: 'Urinary tract infection', 35: 'Psoriasis',36:'Covid19'}   # dictionary


def get_predicted_value(patient_symptoms):  # creating the function  to ge the predicted values for patient symptoms
    input_vector = np.zeros(len(symptoms_dict))  ##This line creates a NumPy array (specifically, a one-dimensional array) filled with zeros.
                                                 #len(symptoms_dict): This part calculates the length (number of elements) of the symptoms_dict.
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1             #symptoms_dict = {'fever': 0, 'cough': 1, 'fatigue': 2}
                                                          #input_vector = [0, 0, 0] (initially all zeros)
                                                          #item = 'cough' if the items is tends to cough the input vector value be[0,1,0]
    return diseases_list[svc.predict([input_vector])[0]] #This part involves making a prediction using a machine learning model (presumably named svc).



#creating routes===============================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        symptoms = request.form.get('symptoms')
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        user_symptoms = [sym.strip("[]'") for sym in user_symptoms]
        predicted_disease = get_predicted_value(user_symptoms)

        desc, pre, med, die, wrkout = helper(predicted_disease)
        my_pre=[]
        for i in pre[0]:
            my_pre.append(i)

        return render_template('index.html',predicted_disease=predicted_disease,dis_desc=desc, dis_pre=my_pre,dis_med=med,dis_wrkout=wrkout,dis_diet=die)



@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/blog')
def blog():
    return render_template('blog.html')
@app.route('/developer')
def developer():
    return render_template('developer.html')

#python main
if __name__=="__main__":
    app.run(debug=True)
