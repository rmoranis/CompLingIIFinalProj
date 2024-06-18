import csv
import os


def process_tsv(input_path, output_path):
    # Read the TSV file
    with open(input_path, 'r') as file:
        print(input_path)
        reader = csv.reader(file, delimiter='\t')
        rows = list(reader)

    # Process the rows
    processed_rows = []
    previous_function = None

    for row in rows:

        # Skip empty rows
        if not row or all(cell == '' for cell in row):
            continue

        # Skip rows containing "=" symbols
        if any('=' in cell for cell in row):
            continue

        # Replace "." in the "function" column with the previous entry
        if len(row) >= 2:
            if row[1].strip() == '.':  # or row[1].strip() == ''
                row[1] = previous_function
            else:
                previous_function = row[1]
        else:  # If there is a chord symbol but no function symbol,
            # treat the function symbol as "."
            if previous_function:
                row.append(previous_function)

        # Remove leading digits (up to two) from the "chords" column
        chords = row[0]
        while chords and (chords[0].isdigit() or chords[0] == "."):
            chords = chords[1:]
        row[0] = chords

        processed_rows.append(row)

    # Write the processed rows to a new TSV file
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(processed_rows)


def process_folder(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process each TSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.tsv'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            process_tsv(input_path, output_path)
            print(f"Processed {filename}")


# Example usage
input_folder = 'outputFiles'
output_folder = 'outputFiles/cleanedFiles'
process_folder(input_folder, output_folder)
