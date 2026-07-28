import os
import re

def convert_files(rel_triples, attr_triples, output):
    with open(rel_triples, "r", encoding="utf-8") as fin, \
            open(output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip().split("\t")
            head = line[0]
            rel = line[1]
            tail = line[2]
            fout.write(f"<{head}> <{rel}> <{tail}> .\n")

    with open(attr_triples, "r", encoding="utf-8") as fin, \
            open(output, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip().split("\t")
            head = line[0]
            rel = line[1]
            if len(line) == 3:
                literal = line[2]
            else:
                literal = ""
            literal = normalize_literal(literal)
            fout.write(f"<{head}> <{rel}> {literal} .\n")


def normalize_literal(line: str) -> str:
    line = line.strip()
    m = re.match(r'^"(.*)"\^\^<(.+)>$', line)
    if m:
        value, dtype = m.groups()
        value = value.replace('\\', '\\\\').replace('"', '\\"')
        return f'"{value}"^^<{dtype}>'
    if line.startswith('\\"') or line.endswith('\\'):
        line = line.strip('\\')
        line = line.strip('"')
    line = line.strip('"')
    line = line.replace('\\', '\\\\').replace('"', '\\"')

    return f'"{line}"'

def convert_openea(base_dir="."):
    for pair_dir in os.listdir(base_dir):
        pair_path = os.path.join(base_dir, pair_dir)

        if not os.path.isdir(pair_path):
            continue

        print(f"Processing dataset: {pair_dir}")

        ent_links = os.path.join(pair_path, "ent_links")
        source_nt = os.path.join(pair_path, "source.nt")
        target_nt = os.path.join(pair_path, "target.nt")
        alignment = os.path.join(pair_path, "alignment")

        convert_files(os.path.join(pair_path, "rel_triples_1"), os.path.join(pair_path, "attr_triples_1"), source_nt)
        convert_files(os.path.join(pair_path, "rel_triples_2"), os.path.join(pair_path, "attr_triples_2"), target_nt)

        with open(ent_links, "r", encoding="utf-8") as fin, \
            open(alignment, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip().split("\t")
                s = line[0]
                t = line[1]
                fout.write(f"<{s}> <{t}>\n")

    print("OpenEA conversion completed.")


if __name__ == "__main__":
    convert_openea()