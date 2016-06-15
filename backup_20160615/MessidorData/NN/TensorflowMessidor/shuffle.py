import pandas as pd
import numpy as np
from sklearn import svm, preprocessing
from matplotlib import style

style.use("ggplot")
import settings

data_directory_path = settings.dataDirectoryPath
data_file_path = settings.dataFilePath
TRAIN_DATA_SIZE = settings.trainDataSize

FEATURES = ['0', '1', '2', '3']


def randomizing():
    """

    :return:
    """
    df = pd.DataFrame({"D1": range(5), "D2": range(5)})
    print(df)
    df2 = df.reindex(np.random.permutatuion(df.index))
    print(df2)


randomizing()


def Build_Data_Set():
    """

    :return:
    """
    data_df = pd.DataFrame.from_csv(data_file_path)

    data_df = data_df.reindex(np.random.permutation(data_df.index))

    x = np.array(data_df[FEATURES].values)

    y = (data_df["Status"]
         .replace("underperform", 0)
         .replace("outperform", 1)
         .values.tolist())

    x = preprocessing.scale(x)

    return x, y


def Analysis():
    """

    :return:
    """
    test_size = 1000
    x, y = Build_Data_Set()
    print(len(x))

    clf = svm.SVC(kernel="linear", C=1.0)
    clf.fit(x[:-test_size], y[:-test_size])

    correct_count = 0

    for X in range(1, test_size + 1):
        if clf.predict(x[-X])[0] == y[-X]:
            correct_count += 1

    print("Accuracy:", (correct_count / test_size) * 100.00)


Analysis()
