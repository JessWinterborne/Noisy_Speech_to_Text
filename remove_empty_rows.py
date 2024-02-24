import csv

def filter_empty_rows(input_csv, output_csv):
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames

        # Write header to the output CSV
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            # Check if the 'text' field is empty
            if row['text'].strip():
                # Write non-empty rows to the output CSV
                writer.writerow(row)

if __name__ == "__main__":
    input_csv_file = '2-3B-ENG-NS.mp3.csv'  # Change this to your input CSV file name
    output_csv_file = '2-3B-ENG-NS.csv'  # Change this to your desired output CSV file name

# filter_empty_rows(input_csv_file, output_csv_file)
