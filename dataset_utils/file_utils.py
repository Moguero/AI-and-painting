import csv
from pathlib import Path
import time

OUTPUT_PATH = Path(r"C:\Users\thiba\PycharmProjects\mission_IA_JCS\deep_learning\test5.csv")
TEST_LIST = range(10)


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
        result = method(*args, **kw)
        end_time = time.time()
        print(f"{method.__name__} : {int(end_time - start_time)}s to execute")
        return result

    return timed
