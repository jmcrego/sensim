import logging
import yaml
import sys
import os
import random
import torch
from shutil import copyfile
from src.tools import create_logger
from src.dataset import Vocab, DataSet, OpenNMTTokenizer
from src.model import make_model
from src.trainer import Trainer

class Argv():

    def __init__(self, argv):
        self.prog = argv.pop(0)
        self.usage = '''usage: {} -dir DIR [-learn YAML] [-infer YAML] [-model YAML] [-optim YAML] [-cuda] [-seed INT] [-log FILE] [-loglevel LEVEL]
   -dir        DIR : checkpoint directory (must not exist when learning from scratch)
   -infer     YAML : test config file (inference mode)
   -learn     YAML : train config file (learning mode)

   -model     YAML : modeling config file (needed when learning from scratch)
   -optim     YAML : optimization config file (needed when learning from scratch)

   -seed       INT : seed value (default 12345)
   -log       FILE : log file (default stderr)
   -loglevel LEVEL : use 'debug', 'info', 'warning', 'critical' or 'error' (default info) 
   -cuda           : use GPU (default not used)
   -h              : this help

* The script needs pyonmttok installed (pip install pyonmttok)
* Use -learn YAML (or -infer YAML) for learning (or inference) modes
* When learning from scratch:
  + The directory -dir DIR is created
  + source and target vocabs/bpe files are copied to DIR (cannot be further modified)
  + config files -optim YAML and -model YAML are copied to DIR (cannot be further modified)
'''.format(self.prog)
        self.fopt = None
        self.fmod = None
        self.flearn = None
        self.flog = None
        self.log_level = 'info'
        self.dir = None
        self.finfer = None
        self.seed = 12345
        self.cuda = False
        while len(argv):
            tok = argv.pop(0)
            if   (tok=="-optim"    and len(argv)): self.fopt = argv.pop(0)
            elif (tok=="-model"    and len(argv)): self.fmod = argv.pop(0)
            elif (tok=="-learn"    and len(argv)): self.flearn = argv.pop(0)
            elif (tok=="-log"      and len(argv)): self.flog = argv.pop(0)
            elif (tok=="-loglevel" and len(argv)): self.log_level = argv.pop(0)
            elif (tok=="-dir"      and len(argv)): self.dir = argv.pop(0)
            elif (tok=="-infer"    and len(argv)): self.finfer = argv.pop(0)
            elif (tok=="-seed"     and len(argv)): self.seed = int(argv.pop(0))
            elif (tok=="-cuda"):                   self.cuda = True
            elif (tok=="-h"):
                sys.stderr.write("{}".format(self.usage))
                sys.exit()
            else:
                sys.stderr.write('error: unparsed {} option\n'.format(tok))
                sys.stderr.write("{}".format(self.usage))
                sys.exit()

        create_logger(self.flog,self.log_level)
                
        if self.dir is None:
            logging.error('error: missing -dir option')
            #sys.stderr.write("{}".format(self.usage))
            sys.exit()

        if self.dir.endswith('/'): 
            self.dir = self.dir[:-1]
        self.dir = os.path.abspath(self.dir)
        logging.info('dir={}'.format(self.dir))

        if self.finfer is not None and self.flearn is not None:
            logging.warning('-learn FILE not used on inference mode')

        if self.finfer is None and self.flearn is None:
            logging.error('either -learn or -infer options must be used')
            sys.exit()

        if self.finfer is not None and not os.path.exists(self.dir):
            logging.error('running inference with empty model dir')
            sys.exit()

        if os.path.exists(self.dir):
            if self.fopt is not None:
                logging.warning('-opt FILE not used ({}/optim.yml)'.format(self.dir))
            if self.fmod is not None:
                logging.warning('-mod FILE not used ({}/model.yml)'.format(self.dir))
            self.fopt = self.dir + "/optim.yml"
            self.fmod = self.dir + "/model.yml"
        else:
            if self.fopt is None:
                logging.error('missing -opt option')
                sys.exit()
            if self.fmod is None:
                logging.error('missing -mod option')
                sys.exit()


def create_experiment(opts):
    with open(opts.fmod) as file:
        mymod = yaml.load(file, Loader=yaml.FullLoader)
        logging.debug('Read model : {}'.format(mymod))
    os.mkdir(opts.dir)
    ### copy vocab/bpe/optim.yml files to opts.dir
    copyfile(mymod['vocab'], opts.dir+"/vocab")
    logging.info('copied {} => {}'.format(mymod['vocab'], opts.dir+"/vocab"))
    copyfile(mymod['tokenization']['src']['bpe_model_path'], opts.dir+"/bpe_model_src")
    logging.info('copied {} => {}'.format(mymod['tokenization']['src']['bpe_model_path'], opts.dir+"/bpe_model_src"))
    copyfile(mymod['tokenization']['tgt']['bpe_model_path'], opts.dir+"/bpe_model_tgt")
    logging.info('copied {} => {}'.format(mymod['tokenization']['tgt']['bpe_model_path'], opts.dir+"/bpe_model_tgt"))
    copyfile(opts.fopt, opts.dir+"/optim.yml")
    logging.info('copied {} => {}'.format(opts.fopt, opts.dir+"/optim.yml"))
    ### update vocb/bpe files in opts.fmod 
    mymod['vocab'] = opts.dir+"/vocab"
    mymod['tokenization']['src']['bpe_model_path'] = opts.dir+"/bpe_model_src"
    mymod['tokenization']['tgt']['bpe_model_path'] = opts.dir+"/bpe_model_tgt"
    ### dump mymod into opts.dir/model.yml
    with open(opts.dir+"/model.yml", 'w') as file:
        _ = yaml.dump(mymod, file)
    ### replace opts.fmod/opts.fopt by the new ones
    opts.fmod = opts.dir+"/model.yml"
    opts.fopt = opts.dir+"/optim.yml"


if __name__ == "__main__":
    
    opts = Argv(sys.argv)

    if opts.seed > 0:
        random.seed(opts.seed)
        torch.manual_seed(opts.seed)
        if opts.cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(opts.seed)
        logging.debug('random.seed set to {}'.format(opts.seed))

    if not os.path.exists(opts.dir):
        ### create directory with all config/vocab/bpe files...
        ### copying also model.yml and optim.yml which cannot be changed anymore
        create_experiment(opts)

    with open(opts.dir+"/model.yml") as file:
        opts.mod = yaml.load(file, Loader=yaml.FullLoader)
        logging.debug('Read model : {}'.format(opts.mod))

    with open(opts.dir+"/optim.yml") as file:
        opts.opt = yaml.load(file, Loader=yaml.FullLoader)
        logging.debug('Read optim : {}'.format(opts.opt))
        
    if opts.finfer is not None:
        with open(opts.finfer) as file:
            opts.test = yaml.load(file, Loader=yaml.FullLoader)
            logging.debug('Read config for inference : {}'.format(opts.test))
    else:
        with open(opts.flearn) as file:
            opts.train = yaml.load(file, Loader=yaml.FullLoader)
            logging.debug('Read config for learning : {}'.format(opts.train))
        trainer = Trainer(opts)
        trainer()

    logging.info('Done!')





