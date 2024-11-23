import os
import re
from pypdf import PdfReader

def parse_pdf_2(pdf_path):
    reader = PdfReader(pdf_path)
    number_of_pages = len(reader.pages)
    text = ""
    for i in range(number_of_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text

def save_text_to_file(text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(text)

def process_pdfs_in_folder(folder_path):
    # Ensure the output directory exists
    output_path = '/Users/siddharthcv/Downloads/courses/Fall 2024/Cloud Computing/paper readings/to_parse'
    os.makedirs(output_path, exist_ok=True)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            try:
                pdf_path = os.path.join(folder_path, filename)
                parsed_text = parse_pdf_2(pdf_path)

                # Clean special characters from filename
                cleaned_filename = re.sub(r'[^A-Za-z0-9]', '_', os.path.splitext(filename)[0])
                output_text_file = os.path.join(output_path, f"{cleaned_filename}.txt")

                # Save parsed text to file
                save_text_to_file(parsed_text, output_text_file)
                print(f"Parsed text has been saved to {output_text_file}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Usage
folder_path = '/Users/siddharthcv/Downloads/courses/Fall 2024/Cloud Computing/paper readings/to_parse'
if os.path.exists(folder_path):
    process_pdfs_in_folder(folder_path)
else:
    print(f"The folder path '{folder_path}' does not exist.")
