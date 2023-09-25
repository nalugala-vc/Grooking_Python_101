import pandas as pd
import re
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

def load_data():
    data = pd.read_excel("Test Files.xlsx")
    return data

def generateEmails(data, email_counts=None):
    if email_counts is None:
        email_counts = {}

    for index, row in data.iterrows():
        student_name = row["Student Name"]

        base_email = student_name[0].lower() + student_name.split(",")[1].split()[-1].lower()

        '''Remove special characters'''
        base_email = re.sub(r'[^a-zA-Z0-9]', '', base_email)

        '''Ensure uniqueness'''
        count = email_counts.get(base_email, 0)
        count += 1
        email_counts[base_email] = count

        if count > 1:
            email = base_email + str(count) + "@gmail.com"
        else:
            email = base_email + "@gmail.com"

        data.at[index, 'Email Address'] = email

    data.to_excel("Test Files.xlsx", index=False)

def separate_genders(data):
    female_data = data[data["Gender"] == "F"]
    male_data = data[data["Gender"] == "M"]

    female_data.to_excel("Female_Emails.xlsx", index=False)
    male_data.to_excel("Male_Emails.xlsx", index=False)

    print("Number of Female Students:", len(female_data))
    print("Number of Male Students:", len(male_data))

def special_names(data):
    pattern = r"\w+'\w+"
    special_names_data = data[data["Student Name"].str.match(pattern)]
    special_names_var = special_names_data["Student Name"].tolist()
    return special_names_var

def similarity_check(data):
    '''Load LaBSE model'''
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    '''Function to embed a list of names'''
    def embed_names(names):
        embeddings = model.encode(names, convert_to_tensor=True)
        return embeddings

    male_names = data[data['Gender'] == 'M']["Student Name"]
    female_names = data[data['Gender'] == 'F']["Student Name"]

    '''Embed male and female names'''
    male_embeddings = embed_names(male_names.tolist())
    female_embeddings = embed_names(female_names.tolist())

    '''similarity matrix'''
    similarity_matrix = util.pytorch_cos_sim(male_embeddings, female_embeddings).numpy()

    '''Filter pairs with at least 50% similarity'''
    similar_pairs = []
    for i in range(len(male_names)):
        for j in range(len(female_names)):
            similarity_value = similarity_matrix[i, j].item()
            if similarity_value >= 0.5:
                similar_pairs.append(
                    {"male_name": male_names.iloc[i], "female_name": female_names.iloc[j], "similarity": similarity_value})

    with open("similar_names.json", "w") as json_file:
        json.dump(similar_pairs, json_file, indent=4, default=float)

def process_students(data):
    with open("similar_names.json", "r") as json_file:
        similar_pairs = json.load(json_file)

    all_student_data = []

    special_names_list = special_names(data)

    for index, row in data.iterrows():
        student_number = row["Student Number"]
        gender = row["Gender"]
        dob = datetime.strptime(str(row["DoB"]), "%Y-%m-%d %H:%M:%S")

        student_name = row["Student Name"]
        has_special_character = student_name in special_names_list
        has_name_similarity = any(pair["male_name"] == student_name or pair["female_name"] == student_name for pair in similar_pairs)

        student_data = {
            "id": index,
            "student_number": student_number,
            "additional_details": {
                "dob": dob.strftime("%Y-%m-%d"),
                "gender": gender,
                "special_character": "yes" if has_special_character else "no",
                "name_similar": "yes" if has_name_similarity else "no"
            }
        }
        all_student_data.append(student_data)

    with open("student_data.json", "w") as json_file:
        json.dump(all_student_data, json_file, indent=4)
