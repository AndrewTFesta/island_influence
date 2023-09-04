"""
@title

@description

"""
import argparse
import json
from pathlib import Path

from island_influence import project_properties


def main(main_args):
    chat_name = 'drafting_gaurav.md'
    chat_fname = Path(project_properties.doc_dir, 'notes', 'drafting_gaurav.md')
    save_path = Path(project_properties.doc_dir, 'notes', f'{Path(chat_name).stem}.json')
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True, parents=True)

    names = ['kili', 'sirius']
    with open(chat_fname, 'r') as chat_file:
        raw_chat = [line.strip() for line in chat_file.readlines() if len(line) > 0]

    chat = {each_name: [] for each_name in names}
    chat['full'] = []
    last_name = None
    for each_line in raw_chat:
        try:
            end_time_idx = each_line.index(']')
            each_line = each_line[end_time_idx + 1:]
            for each_name in names:
                if each_line.startswith(each_name):
                    last_name = each_name
                    each_line = each_line[len(each_name)+2:]
        except ValueError:
            pass

        chat[last_name].append(each_line)
        chat['full'].append((last_name, each_line))

    with open(save_path, 'w') as save_file:
        json.dump(chat, save_file, indent=2)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
