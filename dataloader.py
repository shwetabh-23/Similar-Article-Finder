
def Dataloader(src_path, tgt_path):
    summary = []
    data = []
    with open(src_path, 'r', encoding='utf-8') as f:
        with open(tgt_path, 'r', encoding='utf-8') as f_n:
            for i in range(10000):
                d = f.readline()
                s = f_n.readline()
                summary.append(s)
                data.append(d)
    return data, summary