import csv

input_path = 'description.csv'
output_path = 'description_fixed.csv'

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    header = next(reader)
    writer.writerow(header)
    for row in reader:
        # Gabungkan field jika jumlah kolom > 5 (karena koma di lirik/terjemahan)
        if len(row) > 5:
            # Asumsi: kolom ke-0,1,2 benar, sisanya gabung jadi lirik dan terjemahan
            fixed_row = row[:3]
            # Gabung lirik (kolom ke-3 sampai -2)
            lirik = ','.join(row[3:-1]).replace('\n', '\n')
            terjemahan = row[-1]
            fixed_row.append(lirik)
            fixed_row.append(terjemahan)
            writer.writerow(fixed_row)
        elif len(row) == 5:
            writer.writerow(row)
        else:
            print(f'Baris rusak/kurang kolom: {row}')
print('Selesai! File hasil: description_fixed.csv') 