def convert_to_utf8(input_file, output_file):
    with open(input_file, 'r', encoding='latin-1') as infile:
        content = infile.read()
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


input_file = './indexedFiles/woodbridge.txt'
output_file = './woodbridge_utf8.txt'

convert_to_utf8(input_file, output_file)
