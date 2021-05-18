
from functools import partial
from contextlib import contextmanager
from multiprocessing import Pool

@contextmanager
def poolcontext(*args, **kwargs):
    pool = Pool(*args, **kwargs)
    yield pool
    pool.terminate()

def multi_preprocess(f, tokenizer,file_pathes,featurizer,domain_tokenizer, NCORE):
    with poolcontext(processes=NCORE) as pool:
        output = pool.map(partial(f,tokenizer=tokenizer,featurizer=featurizer,domain_tokenizer=domain_tokenizer), file_pathes)
    ord_data_list = []
    for ii in range(len(output)):
        ord_data_list.append(output[ii])

    return ord_data_list
