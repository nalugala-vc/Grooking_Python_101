from functions import separate_genders, generateEmails , load_data, similarity_check, process_students,special_names

'''load data'''
data = load_data()

'''Generate emails'''
generateEmails(data)

'''Separate genders'''
separate_genders(data)

'''List of students with special characters in name'''
names = special_names(data)
print(names)

'''Check similarity of names'''
similarity_check(data)

'''Generate refined JSON File'''
process_students(data)