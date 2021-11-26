import os
from collections import defaultdict


def get_maps(fp):

    train_kb_fp = os.path.join(fp, 'train.txt')
    entity_map = {}
    relation_map = {}

    with open(train_kb_fp, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [_.split() for _ in lines]
        for line in lines:
            if line[0] not in entity_map:
                entity_map[line[0]] = len(entity_map)
            if line[2] not in entity_map:
                entity_map[line[2]] = len(entity_map)
            if line[1] not in relation_map:
                relation_map[line[1]] = len(relation_map)

    entity_map["<OOV>"] = len(entity_map)
    relation_map["<OOV>"] = len(relation_map)
    return entity_map, relation_map


def get_graph(fp, entity_map, relation_map):

    train_kb_fp = os.path.join(fp, 'train.txt')
    valid_kb_fp = os.path.join(fp, 'valid.txt')
    test_kb_fp = os.path.join(fp, 'test.txt')

    knowledge_graph = defaultdict(lambda: defaultdict(set))

    for filename in [train_kb_fp, valid_kb_fp, test_kb_fp]:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [_.split() for _ in lines]
            for line in lines:
                s = entity_map.get(line[0], entity_map["<OOV>"])
                r = relation_map.get(line[0], relation_map["<OOV>"])
                o = entity_map.get(line[2], entity_map["<OOV>"])
                knowledge_graph[s][r].add(o)

    return knowledge_graph


def process_kbc(fp, entity_map, relation_map, knowledge_graph, mode):

    fp = os.path.join(fp, mode+'.txt')
    facts = []
    facts_filter = []

    with open(fp, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [_.split() for _ in lines]
        for line in lines:
            s = entity_map.get(line[0], entity_map["<OOV>"])
            r = relation_map.get(line[0], relation_map["<OOV>"])
            o = entity_map.get(line[2], entity_map["<OOV>"])
            facts.append([s, r, o])
            facts_filter.append(list(knowledge_graph[s][r]))

    rel_path_ids = [-1 for fact in facts]

    return facts, facts_filter, rel_path_ids


def process_regex(f, entity_map, relation_map, query_type, mode):

    fp = os.path.join(f, 'regex_'+mode)
    fp_filter = os.path.join(f, 'regex_'+mode+'_filter')
    facts = []
    facts_filter = []

    with open(fp, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lines = [_.strip().split('\t') for _ in lines]
        for i in range(0, len(lines)):
            # fact = [type_info, e1, r1, .....rk, e2]
            if int(lines[i][-1]) == query_type:
                # fact = [int(lines[i][-1])]
                fact = [entity_map.get(lines[i][0], entity_map["<OOV>"])]
                fact.extend([relation_map.get(i, relation_map["<OOV>"])
                             for i in lines[i][1:-2]])
                fact.append(entity_map.get(lines[i][-2], entity_map["<OOV>"]))
                facts.append(fact)

    with open(fp_filter) as f:
        lines = f.readlines()
        for line in lines:
            filter_set = []
            if int(line.strip().split('\t')[-1]) == query_type:
                for item in line.strip().split('\t')[: -1]:
                    if not len(item) == 0:
                        filter_set.append(entity_map.get(
                            item, entity_map["<OOV>"]))
                assert len(filter_set) > 0
                facts_filter.append(filter_set)

    rel_path_ids = [-1 for fact in facts]

    return facts, facts_filter, rel_path_ids
