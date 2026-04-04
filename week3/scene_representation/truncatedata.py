import csv

def truncate_csv(input_path, output_path, limit=2000):
    with open(input_path, "r") as infile:
        reader = list(csv.reader(infile))

    header = reader[0]
    rows = reader[1:limit+1]

    with open(output_path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved {output_path}")

# path
dict_path = "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/dict_CLEVR_train_scenes.csv"
hist_path = "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/hist_CLEVR_train_scenes.csv"

truncate_csv(dict_path,
             "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/2k_dict_CLEVR_train_scenes.csv")

truncate_csv(hist_path,
             "/Users/kathychen/PycharmProjects/KC-Veritas/week3/data/2k_hist_CLEVR_train_scenes.csv")