import logging
import sys
import io
import faiss
import numpy as np

def IndexDB(file, d):
	if file.endswith('.gz'): 
		f = gzip.open(fsrc, 'rb')
	else:
		f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

	db = []
	for l in f:
		l = l.rstrip().split(' ')
		if len(l) != d:
			logging.error('found {} floats instead of {}'.format(len(l),d))
			sys.exit()
		db.append(l)

	index = faiss.IndexFlatIP(d)  		# build the index L2
	db = np.array(db).astype('float32')
	index.add(db) # add vectors to the index
	logging.info("read {} vectors".format(index.ntotal))
	return index

def Query(index,file,d,k):
	if file.endswith('.gz'): 
		f = gzip.open(fsrc, 'rb')
	else:
		f = io.open(file, 'r', encoding='utf-8', newline='\n', errors='ignore')

	db = []
	for l in f:
		l = l.rstrip().split(' ')
		if len(l) != d:
			logging.error('found {} floats instead of {}'.format(len(l),d))
			sys.exit()
		db.append(l)

	x = np.array(db).astype('float32')
	D, I = index.search(x, k)
	for i in range(len(D)):
		#print("{}\t{}\t{}".format(i,I[i,0],D[i,0]))
		print("{}\t{}".format(i,I[i]))


if __name__ == '__main__':

	fdb = None
	fquery = None
	d = 512
	k = 10
	verbose = False
	name = sys.argv.pop(0)
	usage = '''usage: {} [-d INT] [-k INT] [-v]
	-db    FILE : file to index
	-query FILE : file with query
	-d      INT : vector size (default 512)
	-k      INT : k-best to retrieve (default 10)
	-v          : verbose output (default False)
	-h          : this help
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
		elif tok=="-query" and len(sys.argv):
			fquery = sys.argv.pop(0)
		elif tok=="-k" and len(sys.argv):
			k = int(sys.argv.pop(0))
		elif tok=="-d" and len(sys.argv):
			d = int(sys.argv.pop(0))
		else:
			sys.stderr.write('error: unparsed {} option\n'.format(tok))
			sys.stderr.write("{}".format(usage))
			sys.exit()


	if fdb is not None:
		indexdb = IndexDB(fdb,d)

	if fquery is not None:
		Query(indexdb,fquery,d,k)

