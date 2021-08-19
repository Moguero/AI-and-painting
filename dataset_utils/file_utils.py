import csv
from pathlib import Path
import time

from loguru import logger

OUTPUT_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\deep_learning\test5.csv")
TEST_LIST = range(10)
PATCHES_COVERAGE_PATH = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files\temp_files\patches_coverage.csv")
DATA_DIR_ROOT = Path(r"C:\Users\thiba\OneDrive - CentraleSupelec\Mission_JCS_IA_peinture\files")


def save_list_to_csv(list_to_export: list, output_path: Path) -> None:
    assert str(output_path)[-4:] == ".csv", f"Specified output path {output_path} is not in format .csv"
    with open(str(output_path), "w", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list_to_export)


def save_dict_to_csv(dict_to_export: dict, output_path: Path) -> None:
    assert str(output_path)[-4:] == ".csv", f"Specified output path {output_path} is not in format .csv"
    with open(str(output_path), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for key, value in dict_to_export.items():
            writer.writerow([key, value])


def load_saved_list(input_path: Path) -> list:
    saved_list = list()
    with open(str(input_path), "r") as f:
        data = csv.reader(f)
        for row in data:
            saved_list += row
    return saved_list


def load_saved_dict(input_path: Path) -> dict:
    saved_dict = dict()
    with open(str(input_path), "r") as f:
        data = csv.reader(f)
        for row in data:
            saved_dict[row[0]] = row[1]
    return saved_dict


def timeit(method):
    """Decorator to time the execution of a function."""

    def timed(*args, **kw):
        start_time = time.time()
        logger.info(f"\nStarting execution of {method.__name__}.")
        result = method(*args, **kw)
        end_time = time.time()
        n_seconds = int(end_time - start_time)
        if n_seconds < 60:
            logger.info(f"\n{method.__name__} : {n_seconds}s to execute")
        elif 60 < n_seconds < 3600:
            logger.info(f"\n{method.__name__} : {n_seconds // 60}min {n_seconds % 60}s to execute")
        else:
            logger.info(f"\n{method.__name__} : {n_seconds // 3600}h {n_seconds % 3600 // 60}min {n_seconds // 3600 % 60}s to execute")
        return result

    return timed


def get_formatted_time():
    return time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

# todo: delete this debug part
# debug
# def f():
#     with open(DATA_DIR_ROOT / "test.csv", "w", newline="") as f:
#         my_list = [{'id': "one", 'a': 1, 'b': [1, 2, 3]}, {'id': 'two', 'a': 3, 'b': [4, 5, 6]}]
#         writer = csv.DictWriter(f, fieldnames=list(my_list[0].keys()))
#         writer.writeheader()
#         for data in my_list:
#             # breakpoint()
#             writer.writerow(data)
#
#
# def g():
#     my_list = list()
#     with open(DATA_DIR_ROOT / 'test.csv', 'r') as csv_file:
#         reader = csv.reader(csv_file)
#         my_list = list(reader)
#     return my_list
#
#
# def h():
#     my_dict = dict()
#     with open(DATA_DIR_ROOT / 'test.csv') as csv_file:
#         reader = csv.reader(csv_file)
#         for row in reader:
#             pass
#     return my_dict
