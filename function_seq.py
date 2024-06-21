import os
import csv
from statistics import mean


def count_consecutive_characters(file_path):
    with open(file_path, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # Skip the header row
        next(reader)

        # Initialize the result list and variables for counting
        result = []
        current_char = None
        count = 0

        for row in reader:
            char = row[1]  # Element in second column only
            if char == current_char:
                count += 1
            else:
                if current_char is not None:
                    result.append(count)
                current_char = char
                count = 1

        # Append the last count
        if count > 0:
            result.append(count)

    return result


def process_folder(folder_path):
    all_counts = []
    file_counts = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(folder_path, file_name)
            counts = count_consecutive_characters(file_path)
            file_counts[file_name] = counts
            all_counts.extend(counts)

    return file_counts, all_counts


def calculate_statistics(counts):
    if not counts:
        return {'average': 0, 'min': 0, 'max': 0}

    avg = mean(counts)
    min_val = min(counts)
    max_val = max(counts)

    return {'average': avg, 'min': min_val, 'max': max_val}


folder_path = 'allFiles'
file_counts, all_counts = process_folder(folder_path)
statistics = calculate_statistics(all_counts)

for file_name, counts in file_counts.items():
    print(f"{file_name}: {counts}")

print(f"All counts: {all_counts}")
print(f"Average: {statistics['average']}")
print(f"Minimum: {statistics['min']}")
print(f"Maximum: {statistics['max']}")
