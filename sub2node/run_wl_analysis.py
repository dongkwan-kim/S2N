import time
from collections import Counter, defaultdict
from pprint import pprint

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import LinearSVC
from termcolor import cprint

from data import get_subgraph_datamodule_for_test
import warnings


warnings.filterwarnings("ignore")


def test(**kwargs):
    NAME = "PPIBP"
    # WLKSRandomTree
    # PPIBP, HPOMetab, HPONeuro, EMUser
    # Density, CC, Coreness, CutRatio
    SUBGRAPH_BATCHING = "connected"  # separated, connected

    WL_STEP = 4
    if NAME.startswith("WL"):
        y_range = range(WL_STEP)
    else:
        y_range = [None]

    lines_to_print = []
    for x_wl_step in range(WL_STEP):
        _sdm = get_subgraph_datamodule_for_test(
            name=NAME,
            use_s2n=False,
            wl4pattern_args=[x_wl_step, "color"],  # color, cluster
            subgraph_batching=SUBGRAPH_BATCHING,
            replace_x_with_wl4pattern=True,
            **kwargs
        )
        for y_idx in y_range:
            if y_idx is not None:
                train_y = _sdm.train_data.y[:, y_idx].numpy()
                val_y = _sdm.val_data.y[:, y_idx].numpy()
                test_y = _sdm.test_data.y[:, y_idx].numpy()
            else:
                train_y = _sdm.train_data.y.numpy()
                val_y = _sdm.val_data.y.numpy()
                test_y = _sdm.test_data.y.numpy()
            train_x = _sdm.train_data.x.numpy()
            val_x = _sdm.val_data.x.numpy()
            test_x = _sdm.test_data.x.numpy()

            all_x = np.concatenate([train_x, val_x, test_x])
            all_y = np.concatenate([train_y, val_y, test_y])
            assert all_x.shape[0] == all_y.shape[0], f"{all_x.shape} != {all_y.shape}"

            try:
                major_class_ratio_test = round(max(Counter(test_y).values()) / test_y.shape[0], 5)
            except TypeError:
                major_class_ratio_test = "N/A"

            # f1_score
            total_trials = 3
            test_f1_list = [None for _ in range(total_trials)]
            # for C in [10 ** 3, 10 ** 2, 10 ** 1, 10 ** 0, 10 ** -1, 10 ** -2, 10 ** -3]:
            for num_trial in range(total_trials):
                best_val_f1 = 0
                for C in [1.0]:
                    model = LinearSVC(random_state=num_trial, C=C)
                    if NAME == "HPONeuro":
                        model = MultiOutputClassifier(model)
                    model.fit(train_x, train_y)
                    val_f1 = f1_score(val_y, model.predict(val_x).astype(int), average="micro")
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        test_f1_list[num_trial] = f1_score(test_y, model.predict(test_x).astype(int), average="micro")
            print(test_f1_list)
            test_f1 = f"{round(np.mean(test_f1_list), 5)} +- {round(np.std(test_f1_list), 5)}"

            # cca
            n_components = 1
            model = CCA(n_components=n_components)
            if NAME != "HPONeuro":
                all_y = np.eye(np.max(all_y) + 1)[all_y]
            all_x_c, all_y_c = model.fit_transform(all_x, all_y)
            corrs = [np.corrcoef(all_x_c[:, i], all_y_c[:, i])[0, 1] for i in range(n_components)]

            lines_to_print.append((f"x={x_wl_step + 1}, y={y_idx}", test_f1, major_class_ratio_test, corrs))
            print(f"x={x_wl_step + 1}, y={y_idx}", test_f1, major_class_ratio_test, corrs)

    print("-" * 22)
    for i, line in enumerate(lines_to_print):
        print("\t".join(str(l) for l in line))
        if (i + 1) % WL_STEP == 0:
            print("-" * 11)


if __name__ == '__main__':
    test()
