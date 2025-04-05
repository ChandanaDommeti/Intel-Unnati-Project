
import pandas as pd
import joblib
import numpy as np
import gradio as gr
import os

print("Initializing Gradio Recommender App with Promotion Prediction...")
SCORE_MODEL_FILE = 'score_predictor_model.pkl'
PROMOTION_MODEL_FILE = 'promotion_predictor_model.pkl' 
LOW_SCORE_THRESHOLD = 60
HIGH_SCORE_THRESHOLD = 90


print(f"Loading SCORE model artifacts from {SCORE_MODEL_FILE}...")
if not os.path.exists(SCORE_MODEL_FILE):
    print(f"FATAL ERROR: Score model file '{SCORE_MODEL_FILE}' not found.")
    print("Please run '008_ScorePredictor.py' first.")
    exit()
try:
    saved_data_score = joblib.load(SCORE_MODEL_FILE)
    score_model = saved_data_score['model']
    score_feature_names_encoded = saved_data_score['feature_names_encoded']
    score_encoders = saved_data_score['encoders']
    score_original_features = saved_data_score['original_features']
    print("Score model artifacts loaded successfully.")
    print(f"  Score Model expects features: {score_feature_names_encoded}")
except Exception as e:
     print(f"FATAL ERROR loading {SCORE_MODEL_FILE}: {e}")
     exit()


print(f"Loading PROMOTION model artifacts from {PROMOTION_MODEL_FILE}...")
if not os.path.exists(PROMOTION_MODEL_FILE):
    print(f"FATAL ERROR: Promotion model file '{PROMOTION_MODEL_FILE}' not found.")
    print("Please run '011_TrainPromotionModel.py' first.")
    exit()
try:
    saved_data_promo = joblib.load(PROMOTION_MODEL_FILE)
    promo_model = saved_data_promo['model']
    promo_feature_names_encoded = saved_data_promo['feature_names_encoded']
    promo_encoders = saved_data_promo['encoders']
    
    promo_target_mapping = saved_data_promo['target_mapping'] 
    print("Promotion model artifacts loaded successfully.")
    print(f"  Promotion Model expects features: {promo_feature_names_encoded}")
    print(f"  Promotion Target Mapping: {promo_target_mapping}")
except Exception as e:
     print(f"FATAL ERROR loading {PROMOTION_MODEL_FILE}: {e}")
     exit()


curriculum = {
    'Math': [
        {'module_id': 'M101', 'name': 'Basic Arithmetic', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []},
        {'module_id': 'M102', 'name': 'Intro to Fractions', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': ['M101']},
        {'module_id': 'M201', 'name': 'Pre-Algebra', 'level': 'Middle School', 'difficulty': 'Medium', 'prereqs': ['M102']},
        {'module_id': 'M202', 'name': 'Algebra I', 'level': 'Middle School', 'difficulty': 'Medium', 'prereqs': ['M201']},
        {'module_id': 'M301', 'name': 'Geometry', 'level': 'High School', 'difficulty': 'Medium', 'prereqs': ['M202']},
        {'module_id': 'M302', 'name': 'Algebra II', 'level': 'High School', 'difficulty': 'Hard', 'prereqs': ['M202']},
        {'module_id': 'M303', 'name': 'Pre-Calculus', 'level': 'High School', 'difficulty': 'Hard', 'prereqs': ['M301', 'M302']},
    ],
    'Science': [
         {'module_id': 'S101', 'name': 'Intro to Biology', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []},
         {'module_id': 'S201', 'name': 'Earth Science', 'level': 'Middle School', 'difficulty': 'Medium', 'prereqs': ['S101']},
         {'module_id': 'S301', 'name': 'Basic Chemistry', 'level': 'High School', 'difficulty': 'Medium', 'prereqs': ['S201', 'M201']},
         {'module_id': 'S302', 'name': 'Basic Physics', 'level': 'High School', 'difficulty': 'Hard', 'prereqs': ['S201', 'M202']},
    ],
    'History': [
        {'module_id': 'H101', 'name': 'Local History', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []},
        {'module_id': 'H201', 'name': 'World History I', 'level': 'Middle School', 'difficulty': 'Medium', 'prereqs': ['H101']},
        {'module_id': 'H301', 'name': 'World History II', 'level': 'High School', 'difficulty': 'Medium', 'prereqs': ['H201']},
    ],
    'Literature': [{'module_id': 'L101', 'name': 'Intro Reading', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []}],
    'Computer Science': [{'module_id': 'CS201', 'name': 'Intro Programming', 'level': 'Middle School', 'difficulty': 'Medium', 'prereqs': []}],
    'Art': [{'module_id': 'A101', 'name': 'Basic Drawing', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []}],
    'Music': [{'module_id': 'MU101', 'name': 'Intro Music', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []}],
}
print("Curriculum map defined.")
available_levels = list(promo_encoders['Level_Student'].classes_) if 'Level_Student' in promo_encoders else ['Elementary', 'Middle School', 'High School']
available_earning_classes = list(promo_encoders['Earning_Class'].classes_) if 'Earning_Class' in promo_encoders else ['Low', 'Middle', 'High']
available_courses = list(score_encoders['Course_Name'].classes_) if 'Course_Name' in score_encoders else list(curriculum.keys())

def find_next_module(student_course, student_level, completed_module_ids=[]):
    if student_course not in curriculum or not curriculum[student_course]:
        return None
    course_modules = curriculum[student_course]
    candidate_module = None
    for module in course_modules:
        if module['level'] == student_level and module['module_id'] not in completed_module_ids:
            prereqs_met = all(prereq in completed_module_ids for prereq in module['prereqs'])
            if prereqs_met: candidate_module = module; break
    if candidate_module is None:
        level_order = ['Elementary', 'Middle School', 'High School']
        try:
            current_index = level_order.index(student_level)
            if current_index + 1 < len(level_order):
                next_level = level_order[current_index + 1]
                for module in course_modules:
                     if module['level'] == next_level and module['module_id'] not in completed_module_ids:
                        prereqs_met = all(prereq in completed_module_ids for prereq in module['prereqs'])
                        if prereqs_met: candidate_module = module; break
        except ValueError: pass
    return candidate_module



def get_recommendation_and_promotion(age, iq, time_per_day, level_student, earning_class, course_name, completed_modules_str):
    """
    Takes student inputs, predicts PROMOTION status AND score for the next module,
    and returns a combined recommendation string.
    """
    print("\n--- Received Request (Combined) ---")
    print(f"Inputs: Age={age}, IQ={iq}, Time={time_per_day}, Level={level_student}, Earning={earning_class}, Course={course_name}, Completed='{completed_modules_str}'")

    if not all([isinstance(age, (int, float)), isinstance(iq, (int, float)), isinstance(time_per_day, (int, float))]):
        return "Error: Age, IQ, and Time Per Day must be numbers."
    if not level_student or not course_name or not earning_class:
        return "Error: Please select Student Level, Earning Class, and Course Name."
    


    predicted_promotion_status_text = "Error predicting promotion"
    try:
        print("  Preparing features for PROMOTION prediction...")
        promo_input_data = {}
        promo_original_features = ['Age', 'IQ', 'Time_Per_Day', 'Level_Student', 'Earning_Class'] # Match training
        profile_map_promo = { 
            'Age': age, 'IQ': iq, 'Time_Per_Day': time_per_day,
            'Level_Student': level_student, 'Earning_Class': earning_class
        }

        for feature in promo_original_features:
            encoded_feature_name = feature + '_Encoded' if feature in promo_encoders else feature
            value_to_encode = profile_map_promo.get(feature)

            if value_to_encode is None: 
                 print(f"    Promo Warning: Missing value for {feature}. Using 0.")
                 promo_input_data[encoded_feature_name] = 0
                 continue

            if feature in promo_encoders:
                
                encoded_value = promo_encoders[feature].transform([value_to_encode])[0]
                promo_input_data[encoded_feature_name] = encoded_value
            else: 
                promo_input_data[encoded_feature_name] = value_to_encode

        promo_input_df = pd.DataFrame([promo_input_data])
        promo_input_df = promo_input_df[promo_feature_names_encoded] 


        prediction_encoded = promo_model.predict(promo_input_df)[0]
        predicted_promotion_status_text = promo_target_mapping.get(prediction_encoded, "Unknown Status")
        print(f"  Predicted Promotion Status (Encoded): {prediction_encoded} -> Text: {predicted_promotion_status_text}")

    except Exception as e:
        print(f"  ERROR during Promotion Prediction: {e}")
        predicted_promotion_status_text = f"Error during prediction: {e}"

    predicted_score = -1
    next_module = None
    recommendation_action_text = "Error determining next course action."
    try:
        
        completed_module_ids = []
        if completed_modules_str:
            completed_module_ids = [mod_id.strip() for mod_id in completed_modules_str.split(',') if mod_id.strip()]

        
        next_module = find_next_module(course_name, level_student, completed_module_ids)

        if next_module:
            print(f"  Identified next potential module: {next_module['name']} (ID: {next_module['module_id']})")
            print("  Preparing features for SCORE prediction...")
            score_input_data = {}
            profile_map_score = { 
                 'Age': age, 'IQ': iq, 'Time_Per_Day': time_per_day,
                 'Level_Student': level_student, 'Course_Name': course_name,
                 'Material_Level': next_module['difficulty'] 
            }

            for feature in score_original_features: 
                encoded_feature_name = feature + '_Encoded' if feature in score_encoders else feature
                value_to_encode = profile_map_score.get(feature)
                source = "Student Profile/Input" if feature != 'Material_Level' else "Next Module Difficulty"

                if value_to_encode is None:
                    print(f"    Score Warning: Missing value for {feature}. Using 0.")
                    score_input_data[encoded_feature_name] = 0
                    continue

                if feature in score_encoders:
                     
                     encoded_value = score_encoders[feature].transform([value_to_encode])[0]
                     score_input_data[encoded_feature_name] = encoded_value
                else: 
                     score_input_data[encoded_feature_name] = value_to_encode

            score_input_df = pd.DataFrame([score_input_data])
            score_input_df = score_input_df[score_feature_names_encoded] 

           
            predicted_score = score_model.predict(score_input_df)[0]
            print(f"  Predicted score for '{next_module['name']}': {predicted_score:.1f}")

            
            if predicted_score > HIGH_SCORE_THRESHOLD:
                recommendation_action_text = f"[High Performance - Predicted Score > {HIGH_SCORE_THRESHOLD}]\n  - Consider offering an accelerated path or 'test-out' option for this module.\n  - If successful, student may be ready for the subsequent module."
            elif predicted_score < LOW_SCORE_THRESHOLD:
                recommendation_action_text = f"[Potential Challenge - Predicted Score < {LOW_SCORE_THRESHOLD}]\n  - Recommend reviewing prerequisite concepts before starting.\n"
                prereqs = next_module.get('prereqs', [])
                if prereqs: recommendation_action_text += f"    - Relevant prerequisites: {', '.join(prereqs)}\n"
                recommendation_action_text += f"  - Suggest providing supplementary materials (e.g., 'Easy' difficulty version) or extra support."
            else:
                recommendation_action_text = f"[Standard Pace - Predicted Score {LOW_SCORE_THRESHOLD}-{HIGH_SCORE_THRESHOLD}]\n  - Proceed with the standard '{next_module['difficulty']}' material for this module.\n  - Monitor progress."

        else: 
             recommendation_action_text = "No suitable next module found based on current curriculum map and completed modules. Cannot predict score or recommend course action."
             print("  No suitable next module found.")

    except Exception as e:
        print(f"  ERROR during Score Prediction / Recommendation Action generation: {e}")
        recommendation_action_text = f"Error during score prediction/recommendation: {e}"


    final_output = f"--- Overall Assessment & Recommendation ---\n\n"
    final_output += f"**Predicted Promotion Status:** {predicted_promotion_status_text.upper()}\n"
    final_output += f"   (Based on general profile: Age, IQ, Time, Level, Earning Class)\n"
    final_output += "-------------------------------------------\n\n"

    final_output += f"**Next Module Recommendation:**\n\n"
    if next_module:
        final_output += f"Proposed Next Module:\n"
        final_output += f"  - Name: '{next_module['name']}'\n"
        final_output += f"  - ID: {next_module['module_id']}\n"
        final_output += f"  - Difficulty: {next_module['difficulty']}\n\n"
        if predicted_score >= 0:
            final_output += f"AI Prediction for this Module:\n"
            final_output += f"  - Predicted Assessment Score: {predicted_score:.1f}\n\n"
        else:
             final_output += f"AI Prediction for this Module: Error predicting score.\n\n"
        final_output += f"Recommended Action:\n{recommendation_action_text}\n"
    else:
        final_output += recommendation_action_text 

    final_output += "\n-------------------------------------------\n"
    print("  Combined recommendation generated.")
    return final_output



print("\nSetting up Gradio interface...")
input_components = [
    gr.Number(label="Student Age", value=12),
    gr.Number(label="Student IQ (Estimated)", value=105),
    gr.Number(label="Daily Study Time (Minutes)", value=60),
    gr.Dropdown(choices=available_levels, label="Current Student Level", value='Middle School'),
    gr.Dropdown(choices=available_earning_classes, label="Parent Earning Class", value='Middle'), 
    gr.Dropdown(choices=available_courses, label="Current Course", value='Math'),
    gr.Textbox(label="Completed Module IDs (Comma-separated, e.g., M101,M102)", placeholder="Leave blank if none")
]

output_component = gr.Textbox(label="AI Assessment & Recommendation", lines=20) 

interface = gr.Interface(
    fn=get_recommendation_and_promotion, 
    inputs=input_components,
    outputs=output_component,
    title="AI K-12 Tutor: Promotion & Course Recommendation",
    description="Enter student details to predict overall promotion status AND get a recommendation for the next course module based on predicted performance.",
    allow_flagging='never',
     examples=[ 
        [10, 95, 45, "Elementary", "Low", "Math", ""],
        [14, 115, 90, "Middle School", "Middle", "Science", "S101"],
        [16, 125, 120, "High School", "High", "Math", "M101,M102,M201,M202"],
        [13, 100, 50, "Middle School", "Middle", "History", "H101"],
        [17, 90, 150, "High School", "Low", "Math", "M101,M102,M201"], 
     ] )

print("Launching Gradio app...")

interface.launch()

print("Gradio app has been launched. Access it via the provided URL.")