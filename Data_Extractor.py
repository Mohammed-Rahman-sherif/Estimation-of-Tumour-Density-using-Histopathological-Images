import glob
import os
import json
import pandas as pd


def importData():
    all_files = glob.glob(
        os.path.join('NuCLS data\\PsAreTruth_E-20211020T171415Z-001-F3\\PsAreTruth_E\\contours\\new',
                     "*.csv"))  # advisable to use os.path.join as this makes concatenation OS independent
    df_from_each_file = pd.DataFrame()
    tempDict = {}
    for f in all_files:
        csv = pd.read_csv(f)

        # Update dictionary with a new object as the filename, then each object will have a tumor_count which counts
        # instances of a tumor, and the total entry count.
        tempDict.update(
            {f: {'tumor_count': int((csv["group"] == "tumor").sum()), 'total_entries': int(csv["group"].count())}})

        df_from_each_file.append(pd.read_csv(f))

    # output to JSON file "test.json"
    with open("test.json", "w") as output:
        json.dump(tempDict, output, indent=4, sort_keys=False)
        output.close()


if __name__ == '__main__':
    importData()
