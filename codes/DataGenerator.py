
import pandas as pd
import numpy as np

print("Starting data generation...")

np.random.seed(42)

N_STUDENTS = 1000
COUNTRIES = ['USA', 'India', 'UK', 'Canada', 'Australia']
STATES = {
    'USA': ['California', 'Texas', 'New York', 'Florida', 'Illinois'],
    'India': ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Uttar Pradesh'],
    'UK': ['England', 'Scotland', 'Wales', 'Northern Ireland'],
    'Canada': ['Ontario', 'Quebec', 'British Columbia', 'Alberta'],
    'Australia': ['NSW', 'Victoria', 'Queensland', 'Western Australia']
}
CITIES = {
    'California': ['Los Angeles', 'San Francisco', 'San Diego'],
    'Texas': ['Houston', 'Dallas', 'Austin'],
    'New York': ['New York City', 'Buffalo'],
    'Florida': ['Miami', 'Orlando'],
    'Illinois': ['Chicago'],
    'Maharashtra': ['Mumbai', 'Pune', 'Nagpur'],
    'Karnataka': ['Bangalore', 'Mysore'],
    'Tamil Nadu': ['Chennai', 'Coimbatore'],
    'Delhi': ['Delhi'],
    'Uttar Pradesh': ['Lucknow', 'Kanpur'],
    'England': ['London', 'Manchester', 'Birmingham'],
    'Scotland': ['Glasgow', 'Edinburgh'],
    'Wales': ['Cardiff'],
    'Northern Ireland': ['Belfast'],
    'Ontario': ['Toronto', 'Ottawa'],
    'Quebec': ['Montreal', 'Quebec City'],
    'British Columbia': ['Vancouver', 'Victoria'],
    'Alberta': ['Calgary', 'Edmonton'],
    'NSW': ['Sydney'],
    'Victoria': ['Melbourne'],
    'Queensland': ['Brisbane'],
    'Western Australia': ['Perth']
}
OCCUPATIONS = ['Engineer', 'Teacher', 'Doctor', 'Artist', 'Business Owner', 'Service Worker', 'Admin', 'Unemployed', 'Other']
COURSES = ['Math', 'Science', 'History', 'Literature', 'Computer Science', 'Art', 'Music']
STUDENT_LEVELS = ['Elementary', 'Middle School', 'High School']
COURSE_LEVELS = ['Beginner', 'Intermediate', 'Advanced'] 
MATERIAL_LEVELS = ['Easy', 'Medium', 'Hard']

data = {
    'Name': [f'Student_{i}' for i in range(1, N_STUDENTS+1)],
    'Age': np.random.randint(6, 18, N_STUDENTS),
    'Gender': np.random.choice(['Male', 'Female', 'Other'], N_STUDENTS, p=[0.48, 0.48, 0.04]),

    'Country': np.random.choice(COUNTRIES, N_STUDENTS),
    'State': None,
    'City': None, 

    'Parent_Occupation': np.random.choice(OCCUPATIONS, N_STUDENTS, p=[0.15, 0.15, 0.08, 0.05, 0.12, 0.15, 0.1, 0.1, 0.1]),
    'Earning_Class': np.random.choice(['Low', 'Middle', 'High'], N_STUDENTS, p=[0.35, 0.55, 0.1]),

    'Level_Student': None,
    'Level_Course': np.random.choice(COURSE_LEVELS, N_STUDENTS),
    'Course_Name': np.random.choice(COURSES, N_STUDENTS),
    'Time_Per_Day': np.random.randint(20, 240, N_STUDENTS), 
    'Material_Level': np.random.choice(MATERIAL_LEVELS, N_STUDENTS, p=[0.3, 0.5, 0.2]),
    'IQ': np.clip(np.random.normal(100, 15, N_STUDENTS).astype(int), 70, 145),

    'Assessment_Score': None, 
}

df = pd.DataFrame(data)

df['State'] = df['Country'].apply(lambda x: np.random.choice(STATES.get(x, ['Unknown State'])))
df['City'] = df.apply(lambda row: np.random.choice(CITIES.get(row['State'], ['Unknown City'])), axis=1)

def assign_level(age):
    if age <= 10: return 'Elementary'
    if age <= 14: return 'Middle School'
    return 'High School'
df['Level_Student'] = df['Age'].apply(assign_level)

base_score = 70
iq_effect = (df['IQ'] - 100) * 0.3 
time_effect = (df['Time_Per_Day'] - 90) * 0.05 
material_difficulty_map = {'Easy': -5, 'Medium': 0, 'Hard': 5} 
material_effect = df['Material_Level'].map(material_difficulty_map)
random_noise = np.random.normal(0, 8, N_STUDENTS) 

df['Assessment_Score'] = base_score + iq_effect + time_effect - material_effect + random_noise

df['Assessment_Score'] = df['Assessment_Score'].clip(30, 100).astype(int)

df['Promotion_Status'] = np.where(df['Assessment_Score'] >= 50, 'Yes', 'No')

print("\nGenerated DataFrame Head:")
print(df.head())
print("\nDataFrame Info:")
df.info()
print("\nValue Counts for Key Categories:")
print("Level_Student:\n", df['Level_Student'].value_counts())
print("Material_Level:\n", df['Material_Level'].value_counts())
print("Promotion_Status:\n", df['Promotion_Status'].value_counts())


# Save to CSV
output_filename = 'k12_tutoring_dataset.csv'
df.to_csv(output_filename, index=False)
print(f"\n✅ Dataset generated and saved successfully to: {output_filename}")