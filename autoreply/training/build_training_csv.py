import glob
import csv

# Directory where the TXT files are stored
directory_path = "data"

# Use glob to find all non-zero byte .txt files in the specified directory
txt_files = glob.glob(f"{directory_path}/*.txt")

# Function to parse .txt files and extract the narrative and classifications
def parse_txt_files(file_paths):
    extracted_data = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
            
            # Extract "Generated Text:" content
            generated_text = ""
            for line in lines:
                if line.startswith("Generated Text:"):
                    generated_text = line.replace("Generated Text:", "").strip()
                    #print(generated_text)
                    break
            
            # Extract classifications (collect all lines after "Sentiment Classification:")
            
            sentiment = ""
            for line in lines:
                if line.startswith("Sentiment Classification:"):
                    sentiment = line.replace("Sentiment Classification:", "").strip()
                    
            pacing = ""
            for line in lines:
                if line.startswith("Pacing Assessment:"):
                    pacing = line.replace("Pacing Assessment:", "").strip()  

            plotting = ""
            for line in lines:
                if line.startswith("Plot Dynamics Classification:"):
                    plotting = line.replace("Plot Dynamics Classification:", "").strip()            

            if generated_text != "" and sentiment != "" and pacing != "" and plotting != "":
                extracted_data.append((generated_text, f"{sentiment}, {pacing}, {plotting}"))
    
    return extracted_data

# Parse the non-zero byte .txt files
extracted_data = parse_txt_files(txt_files)

# Define CSV file path for the training data
csv_path = f"{directory_path}/training.csv"

# Write the parsed data to a CSV file
with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["generated_narrative", "classifications"])  # Writing header
    for row in extracted_data:
        writer.writerow(row)

csv_path
