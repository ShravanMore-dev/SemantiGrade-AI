import csv

input_file = "dataset.csv"
output_file = "dataset_fixed.csv"

with open(input_file, newline='', encoding='utf-8') as f:
    rows = list(csv.reader(f))

header, *data_rows = rows
fixed_rows = [header]

for i, row in enumerate(data_rows, start=2):  # start=2 to match file line numbers
    if len(row) == 4:
        fixed_rows.append(row)
    elif len(row) > 4:
        question = row[0]
        human_score = row[-1]
        student_answer = row[-2]
        reference_answer = ",".join(row[1:-2])  # rejoin the split fragments
        fixed_rows.append([question, reference_answer, student_answer, human_score])
        print(f"Fixed line {i}: {len(row)} fields -> 4")
    else:
        print(f"⚠️ Line {i} has only {len(row)} fields, needs manual check: {row}")

with open(output_file, "w", newline='', encoding='utf-8') as f:
    csv.writer(f, quoting=csv.QUOTE_MINIMAL).writerows(fixed_rows)

print(f"\n✅ Wrote {output_file}")