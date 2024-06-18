import os  # os module to interact with the operating system
import csv


# Function to parse Humdrum .krn file and convert it to TSV
def convert_krn_to_tsv(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # Extract header (first line starting with **)
    header = []
    for line in lines:
        if line.startswith('**'):
            header = line.strip().split('\t')
            break

    # Open a TSV file to write the data
    with open(output_file, mode='w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')

        # Write header to TSV
        writer.writerow(header)

        # Write lines of data
        for line in lines:
            if not line.startswith(('!', '*', '=')) and \
                    not line.startswith('**'):
                row = line.strip().split('\t')
                writer.writerow(row)
            elif line.startswith('='):
                # Preserve measure markers
                row = line.strip().split('\t')
                writer.writerow(row)


# Directory containing .krn files for one work by Encoder A,
# with TAVERN available locally as TAVERN-master.
# The directory must be changed to convert the files for other works,
# verifying that Encoder A encoded that work.
input_directory = 'TAVERN-master/Beethoven/B063/Encodings/Encoder_A'
# Output directory
output_directory = 'outputFiles'

# Ensure output directory exists
os.makedirs(output_directory, exist_ok=True)

# Loop over all .krn files in the input directory
for filename in os.listdir(input_directory):  # List all files in input dir
    if filename.endswith('.krn'):  # Check if the file ends with .krn
        # Construct full input file path
        input_file = os.path.join(input_directory, filename)
        # Construct full output file path
        output_file = os.path.join(output_directory,
                                   filename.replace('.krn', '.tsv'))
        # Convert the .krn file to .tsv
        convert_krn_to_tsv(input_file, output_file)
        print(f'Converted {input_file} to {output_file}')
