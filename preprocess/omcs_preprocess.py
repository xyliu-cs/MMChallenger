import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Preprocess the input files.')
parser.add_argument('input_file_path', type=str, help='The path to the input file (OMCS sentences file)')
parser.add_argument('output_file_path', type=str, help='The path to the output file (Filtered English sentences)')

# Execute the parse_args() method
args = parser.parse_args()
input_file_path = args.input_file_path
output_file_path = args.output_file_path

def handle_bad_lines_to_file(line):
    with open('invalid_lines.txt', 'a') as f:
        f.write(f"{line}\n")

# we log the bad sentences that have indention problems and drop them in the dataframe.
df = pd.read_csv(input_file_path, sep='\t', engine='python', on_bad_lines=handle_bad_lines_to_file) # note that if using the 'C' engine the result will be different.
# lines with non-numeric values for their ids
invalid_rows = df[pd.to_numeric(df.iloc[:, 0], errors='coerce').isna() | pd.to_numeric(df.iloc[:, 2], errors='coerce').isna()] 

with open('invalid_lines.txt', 'a') as f:
    for index, row in invalid_rows.iterrows():
        row_as_list = row.tolist()
        f.write(f"{row_as_list}\n")

# drop the invalid rows
df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')
df.iloc[:, 2] = pd.to_numeric(df.iloc[:, 2], errors='coerce')
df.dropna(subset=[df.columns[0], df.columns[2]], inplace=True)

# extract english sentences
filtered_df = df[df['language_id'] == 'en']
sentences = filtered_df['text'].replace('\n', ' ', regex=True)  # some sentences contain '\n' character
en_rows = filtered_df.shape[0]

with open(output_file_path, 'w', encoding='utf-8') as file:
    for sentence in sentences:
        file.write(sentence + '\n') 
print(f"{en_rows} sentences with 'language_id'='en' have been written to {output_file_path}")