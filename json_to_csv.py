import json
import sys

json_file = sys.argv[1]
data = json.load(open(json_file))

print(data["rouge"]["rouge1"], data["rouge"]["rouge2"], data["bleu"]["bleu"], data["bleu"]["precisions"][0], data["bleu"]["precisions"][1], data["meteor"]["meteor"], data["bertscore"]["precision"], data["bertscore"]["recall"], data["bertscore"]["f1"], sep=",")