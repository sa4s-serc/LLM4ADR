import json
import os
import csv

def main():
    folder = 'repositories'
    template_count = {}
    # create new csv file
    with open('data.csv', 'w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['URL', 'ADR Folders', 'Number of ADR Files', 'Names of ADR Folders', 'Names of ADR Files'])
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename)) as json_file:
                data = json.load(json_file)
                
                names_of_folders = '| '.join(data['adrDirectories'])
                names_of_files = '| '.join((adr['adrDirectory'] + '/' + adr['path']) for adr in data['adrFiles'])
                
                for file in data['adrFiles']:
                    if template_count.get(file['template']) is None:
                        template_count[file['template']] = 1
                    else:
                        template_count[file['template']] += 1
                
                writer.writerow([data['repositoryUrl'], data['numAdrDirectories'], data['numAdrFiles'], names_of_folders, names_of_files])
    print(template_count)

if __name__ == '__main__':
    main()