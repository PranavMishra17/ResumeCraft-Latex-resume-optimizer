with open('run_output.txt', 'r', encoding='utf-16le') as f1, open('run_output_utf8.txt', 'w', encoding='utf-8') as f2:
    f2.write(f1.read())
