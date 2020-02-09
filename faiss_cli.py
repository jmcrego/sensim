import logging
import sys
import io
import faiss
import numpy as np

class IndexFaiss:

    def __init__(self, file, d, file_str=None):
        if file.endswith('.gz'): 
            f = gzip.open(file, 'rb')
        else:
            f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

        self.db = []
        for l in f:
            l = l.rstrip().split(' ')
            if len(l) != d:
                logging.error('found {} floats instead of {}'.format(len(l),d))
                sys.exit()
            self.db.append(l)

        self.db_str = []
        if file_str is not None:
            if file_str.endswith('.gz'): 
                f = gzip.open(file_str, 'rb')
            else:
                f = io.open(file_str, 'r', encoding='utf-8', newline='\n', errors='ignore')
            for l in f:
                self.db_str.append(l.rstrip())


    #build an index with METRIC_INNER_PRODUCT
    #normalize the vectors prior to adding them to the index (with faiss.normalize_L2 in Python)
    #normalize the vectors prior to searching them

        self.index = faiss.IndexFlatIP(d)       # build the index L2
        self.db = np.array(self.db).astype('float32')
        self.index.add(faiss.normalize_L2(self.db)) # add vectors to the index
        logging.info("read {} vectors".format(self.index.ntotal))


    def Query(self,file,d,k,file_str):
        if file.endswith('.gz'): 
            f = gzip.open(fsrc, 'rb')
        else:
            f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

        query = []
        for l in f:
            l = l.rstrip().split(' ')
            if len(l) != d:
                logging.error('found {} floats instead of {}'.format(len(l),d))
                sys.exit()
            query.append(l)

        query_str = []
        if file_str is not None:
            if file_str.endswith('.gz'): 
                f = gzip.open(file_str, 'rb')
            else:
                f = io.open(file_str, 'r', encoding='utf-8', newline='\n', errors='ignore')
            for l in f:
                query_str.append(l.rstrip())

        n_ok = [0.0] * k
        x = np.array(query).astype('float32')
        D, I = self.index.search(x, k)
        for i in range(len(I)):
            out = []
            out.append(str(i))
            out.append("{} {}".format(I[i],D[i]))
            if len(query_str):
                out.append(query_str[i])
            if len(self.db_str):
                out.append(self.db_str[I[i,0]])
            print('\t'.join(out))
            ### Accuracy
            for j in range(k):
                if i in I[i,0:j+1]:
                    n_ok[j] += 1.0

        n_ok = ["{:.3f}".format(n/len(x)) for n in n_ok]
        print('Done k-best Acc = {} over {} examples'.format(n_ok,len(x)))

if __name__ == '__main__':

    fdb = None
    fquery = None
    fdb_str = None
    fquery_str = None
    d = 512
    k = 10
    verbose = False
    name = sys.argv.pop(0)
    usage = '''usage: {} -db FILE -query FILE [-db_str FILE] [-query_str] [-d INT] [-k INT] [-v]
    -db        FILE : file to index 
    -db_str    FILE : file to index 
    -query     FILE : file with queries
    -query_str FILE : file with queries
    -d          INT : vector size (default 512)
    -k          INT : k-best to retrieve (default 10)
    -v              : verbose output (default False)
    -h              : this help
'''.format(name)

    while len(sys.argv):
        tok = sys.argv.pop(0)
        if tok=="-h":
            sys.stderr.write("{}".format(usage))
            sys.exit()
        elif tok=="-v":
            verbose = True
        elif tok=="-db" and len(sys.argv):
            fdb = sys.argv.pop(0)
        elif tok=="-db_str" and len(sys.argv):
            fdb_str = sys.argv.pop(0)
        elif tok=="-query" and len(sys.argv):
            fquery = sys.argv.pop(0)
        elif tok=="-query_str" and len(sys.argv):
            fquery_str = sys.argv.pop(0)
        elif tok=="-k" and len(sys.argv):
            k = int(sys.argv.pop(0))
        elif tok=="-d" and len(sys.argv):
            d = int(sys.argv.pop(0))
        elif tok=="-query_is_db":
            query_is_db = True
        else:
            sys.stderr.write('error: unparsed {} option\n'.format(tok))
            sys.stderr.write("{}".format(usage))
            sys.exit()


    if fdb is not None:
        indexdb = IndexFaiss(fdb,d,fdb_str)

    if fquery is not None:
        indexdb.Query(fquery,d,k,fquery_str)

