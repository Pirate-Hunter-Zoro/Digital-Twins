import ijson
import json
import os

# Your combined JSON file path
base_data_path = "/home/librad.laureateinstitute.org/mferguson/Digital-Twins/real_data/" # Your requested output directory
combined_json_output_path = os.path.join(base_data_path, "all_patients_combined_streamed.json")

num_patients_to_read = 5 # How many patient records you want to see

print(f"Attempting to read the first {num_patients_to_read} patients from: {combined_json_output_path}")

try:
    with open(combined_json_output_path, 'r') as f:
        # ijson.items(f, '') means read items from the root of the JSON file.
        # Since your combined JSON is a list of patient objects, '' targets each item in the root list.
        # If it were a JSON object with a key like {"patients": [...]}, you'd use 'patients.item'
        patient_generator = ijson.items(f, '')

        read_count = 0
        for patient in patient_generator:
            print(f"\n--- Patient {read_count + 1} ---")
            # You can print the whole dictionary, or just specific parts
            print(json.dumps(patient, indent=2)[:1000] + "\n..." if len(json.dumps(patient, indent=2)) > 1000 else json.dumps(patient, indent=2)) # Print snippet for large patient JSONs

            read_count += 1
            if read_count >= num_patients_to_read:
                break # Stop after reading the desired number of patients

    if read_count == 0:
        print("No patient data found or file is empty after streaming.")

except FileNotFoundError:
    print(f"Error: The file {combined_json_output_path} was not found.")
except Exception as e:
    print(f"An unexpected error occurred while reading the combined JSON: {e}")