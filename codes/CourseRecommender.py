
import pandas as pd
import joblib
import numpy as np

print("Starting Course Recommender...")

STUDENT_DATA_FILE = 'k12_tutoring_dataset.csv'
MODEL_FILE = 'score_predictor_model.pkl'

LOW_SCORE_THRESHOLD = 60
HIGH_SCORE_THRESHOLD = 90
try:
    saved_data = joblib.load(MODEL_FILE)
    model = saved_data['model']
    feature_names_encoded = saved_data['feature_names_encoded']
    encoders = saved_data['encoders']
    original_features = saved_data['original_features']
    print(f"Loaded model artifacts from {MODEL_FILE}")
    print(f"  Model expects features: {feature_names_encoded}")
except FileNotFoundError:
    print(f"Error: Model file {MODEL_FILE} not found. Run 008_ScorePredictor.py first.")
    exit()
except KeyError as e:
    print(f"Error: Missing key in {MODEL_FILE}. It might be corrupted or saved incorrectly: {e}")
    exit()

try:
    student_df_full = pd.read_csv(STUDENT_DATA_FILE)
    print(f"Loaded student data from {STUDENT_DATA_FILE}")
except FileNotFoundError:
     print(f"Error: {STUDENT_DATA_FILE} not found. Run 007_DataGenerator.py first.")
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
         {'module_id': 'S301', 'name': 'Basic Chemistry', 'level': 'High School', 'difficulty': 'Medium', 'prereqs': ['S201', 'M201']}, # Example cross-subject prereq
         {'module_id': 'S302', 'name': 'Basic Physics', 'level': 'High School', 'difficulty': 'Hard', 'prereqs': ['S201', 'M202']},
    ],
    'History': [
        {'module_id': 'H101', 'name': 'Local History', 'level': 'Elementary', 'difficulty': 'Easy', 'prereqs': []},
        {'module_id': 'H201', 'name': 'World History I', 'level': 'Middle School', 'difficulty': 'Medium', 'prereqs': ['H101']},
        {'module_id': 'H301', 'name': 'World History II', 'level': 'High School', 'difficulty': 'Medium', 'prereqs': ['H201']},
    ],
    'Literature': [],
    'Computer Science': [],
    'Art': [],
    'Music': []
}
print("\nDefined simplified curriculum map.")

def find_next_module(student_course, student_level, completed_module_ids=[]):
    
    if student_course not in curriculum or not curriculum[student_course]:
        print(f"  Warning: Course '{student_course}' not found or empty in curriculum.")
        return None

    course_modules = curriculum[student_course]
    candidate_module = None
    for module in course_modules:
        if module['level'] == student_level and module['module_id'] not in completed_module_ids:
            prereqs_met = all(prereq in completed_module_ids for prereq in module['prereqs'])
            if prereqs_met:
                 candidate_module = module
                 break
    if candidate_module is None:
        level_order = ['Elementary', 'Middle School', 'High School']
        try:
            current_index = level_order.index(student_level)
            if current_index + 1 < len(level_order):
                next_level = level_order[current_index + 1]
                print(f"  No suitable module at {student_level}, checking {next_level}...")
                for module in course_modules:
                     if module['level'] == next_level and module['module_id'] not in completed_module_ids:
                        prereqs_met = all(prereq in completed_module_ids for prereq in module['prereqs'])
                        if prereqs_met:
                            candidate_module = module
                            break
        except ValueError:
            print(f"  Warning: Student level '{student_level}' not in defined order.")

    if candidate_module:
        print(f"  Identified next potential module: {candidate_module['name']} (ID: {candidate_module['module_id']})")
        return candidate_module
    else:
        print(f"  No suitable next module found for course '{student_course}' at level '{student_level}' or above.")
        return None

def recommend_course_action(student_profile, completed_module_ids=[]):
    
    if student_profile is None or student_profile.empty:
        return "Error: Invalid student profile provided."

    current_course = student_profile.get('Course_Name', 'Unknown')
    current_level = student_profile.get('Level_Student', 'Unknown')

    print(f"\nGenerating recommendation for Student {student_profile.get('Name', 'Unknown')} (Course: {current_course}, Level: {current_level})...")
    print(f"  Completed Modules: {completed_module_ids}")

    next_module = find_next_module(current_course, current_level, completed_module_ids)

    if not next_module:
        return "Recommendation: No suitable next module found based on current curriculum map and completed modules. Consider reviewing progress or updating curriculum."

    input_data = {}
    print("  Preparing features for prediction...")
    for feature in original_features:
        encoded_feature_name = feature + '_Encoded' if feature in encoders else feature

        value_to_encode = None
        source = "Unknown"

        
        if feature == 'Material_Level':
        
            value_to_encode = next_module['difficulty']
            source = f"Next Module ({next_module['module_id']})"
        elif feature in student_profile.index:
             value_to_encode = student_profile[feature]
             source = "Student Profile"
        else:
             print(f"    Warning: Feature '{feature}' not found in student profile. Using default 0.")
             
             input_data[encoded_feature_name] = 0 
             continue 

        
        if feature in encoders:
            try:
                
                encoded_value = encoders[feature].transform([value_to_encode])[0]
                input_data[encoded_feature_name] = encoded_value
                print(f"    Feature '{feature}' (Source: {source}): Value='{value_to_encode}' -> Encoded='{encoded_value}'")
            except ValueError:
                
                print(f"    Warning: Value '{value_to_encode}' for feature '{feature}' not seen during training encoder fit. Using -1 as fallback.")
                input_data[encoded_feature_name] = -1 
            except Exception as e:
                print(f"    Error encoding feature '{feature}' with value '{value_to_encode}': {e}")
                input_data[encoded_feature_name] = -1 
        elif pd.api.types.is_numeric_dtype(type(value_to_encode)):
             
             input_data[encoded_feature_name] = value_to_encode 
             print(f"    Feature '{feature}' (Source: {source}): Value='{value_to_encode}' (Numeric, kept as is)")
        else:
             print(f"    Warning: Feature '{feature}' is neither numeric nor in encoders. Using default 0.")
             input_data[encoded_feature_name] = 0

    
    try:
        input_df = pd.DataFrame([input_data])
    
        input_df = input_df[feature_names_encoded]
        print("    Prepared prediction input DataFrame:")
        print(input_df.to_string())
    except KeyError as e:
        print(f"    Error: Mismatch between prepared features and model's expected features. Missing: {e}")
        return "Error: Feature mismatch during prediction preparation. Cannot proceed."
    except Exception as e:
         print(f"    Error creating prediction DataFrame: {e}")
         return "Error: Could not prepare data for prediction."


    
    try:
        predicted_score = model.predict(input_df)[0]
        print(f"  Predicted score for '{next_module['name']}': {predicted_score:.1f}")
    except Exception as e:
        print(f"  Error during model prediction: {e}")
        return f"Error: Could not predict score for module {next_module['name']}."



    recommendation = f"--- Recommendation for Student {student_profile.get('Name', 'Unknown')} ---\n"
    recommendation += f"Next Proposed Module: '{next_module['name']}' (ID: {next_module['module_id']}, Difficulty: {next_module['difficulty']})\n"
    recommendation += f"Predicted Score: {predicted_score:.1f}\n\n"
    recommendation += "Action:\n"

    if predicted_score > HIGH_SCORE_THRESHOLD:
        recommendation += f"[High Performance - Score > {HIGH_SCORE_THRESHOLD}]\n"
        recommendation += f"  - Consider offering an accelerated path or 'test-out' option for '{next_module['name']}'.\n"
        recommendation += f"  - If successful, potentially advance to the module *after* '{next_module['name']}'."
        
        

    elif predicted_score < LOW_SCORE_THRESHOLD:
        recommendation += f"[Potential Challenge - Score < {LOW_SCORE_THRESHOLD}]\n"
        recommendation += f"  - Recommend reviewing prerequisite concepts before starting '{next_module['name']}'.\n"
        prereqs = next_module.get('prereqs', [])
        if prereqs:
             recommendation += f"    - Suggested prerequisites: {', '.join(prereqs)}\n"
        recommendation += f"  - Consider providing supplementary materials at an 'Easy' difficulty for '{next_module['name']}'.\n"
        recommendation += f"  - Offer additional support or examples for this module."
    else:
        recommendation += f"[Standard Pace - Score {LOW_SCORE_THRESHOLD}-{HIGH_SCORE_THRESHOLD}]\n"
        recommendation += f"  - Proceed with the standard '{next_module['difficulty']}' material for '{next_module['name']}'.\n"
        recommendation += f"  - Monitor progress as usual."

    recommendation += "\n-------------------------------------\n"
    return recommendation

if __name__ == "__main__":
    
    try:
    
        student_to_test_index = 10 
        sample_student_profile = student_df_full.iloc[student_to_test_index]
        
        print(f"\nSelected sample student (Index: {student_to_test_index}):\n{sample_student_profile.to_string()}")
    except IndexError:
        print(f"Error: Student at index {student_to_test_index} not found.")
        exit()
    
    print("\n=== SCENARIO 1: Student Starting Course ===")
    completed_modules_scenario1 = []
    recommendation_result_1 = recommend_course_action(sample_student_profile, completed_modules_scenario1)
    print(recommendation_result_1)

   
    print("\n=== SCENARIO 2: After Completing First Module ===")

    first_module = find_next_module(sample_student_profile['Course_Name'], sample_student_profile['Level_Student'], [])
    if first_module:
        completed_modules_scenario2 = [first_module['module_id']]
        print(f"  (Simulating completion of: {first_module['name']} - {first_module['module_id']})")
        recommendation_result_2 = recommend_course_action(sample_student_profile, completed_modules_scenario2)
        print(recommendation_result_2)
    else:
        print("  Could not determine the first module to simulate completion.")


    try:
        student_to_test_index_2 = 5 
        sample_student_profile_2 = student_df_full.iloc[student_to_test_index_2]
        print(f"\n=== SCENARIO 3: Different Student (Index: {student_to_test_index_2}) ===")
        print(f"Selected sample student:\n{sample_student_profile_2.to_string()}")
        recommendation_result_3 = recommend_course_action(sample_student_profile_2, []) 
        print(recommendation_result_3)
    except IndexError:
        print(f"Error: Student at index {student_to_test_index_2} not found.")
    


    print("\n✅ Course recommendation script finished.")