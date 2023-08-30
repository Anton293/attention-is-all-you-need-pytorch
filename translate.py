''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.data import Dataset
from transformer.Models import Transformer
from transformer.Translator import Translator


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 


def main():
    '''Main Function'''

    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    parser.add_argument('-data_pkl', required=True,
                        help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)
    parser.add_argument('-no_cuda', action='store_true')

    # TODO: Translate bpe encoded files 
    #parser.add_argument('-src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    #parser.add_argument('-vocab', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    #parser.add_argument('-batch_size', type=int, default=30,
    #                    help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda

    data = pickle.load(open(opt.data_pkl, 'rb'))
    SRC, TRG = data['vocab']['src'], data['vocab']['trg']
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    
    device = torch.device('cuda' if opt.cuda else 'cpu')
    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example.src]
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
            #print(pred_line)
            f.write(pred_line.strip() + '\n')

    print('[Info] Finished.')


import codecs
from apply_bpe import BPE
import torchtext


# завантажуємо токенізатор bpe
with codecs.open("./output/codes.bpe", encoding='utf-8') as codes:
        bpe = BPE(codes, separator='@@')

data = pickle.load(open("./output/tokenized-data.pkl", 'rb'))
device = torch.device('cpu')
#./datasets/model.chkpt
opt = argparse.Namespace()
opt.model = "./output/model.chkpt"
opt.cuda = "cpu"
opt.beam_size = 4
opt.max_seq_len = 40
opt.no_cuda = False
opt.data_pkl = "./output/tokenized-data.pkl"
opt.output = "pred.txt"
translator = Translator(
        model=load_model(opt, device),
        beam_size=5,
        max_seq_len=data['settings']['max_len'],
        src_pad_idx=data['vocab']['src'].vocab.stoi[Constants.PAD_WORD],
        trg_pad_idx=data['vocab']['trg'].vocab.stoi[Constants.PAD_WORD],
        trg_bos_idx=data['vocab']['trg'].vocab.stoi[Constants.BOS_WORD],
        trg_eos_idx=data['vocab']['trg'].vocab.stoi[Constants.EOS_WORD]
    ).to(device)

def bpe_tokenize(line):
    return bpe.process_line(line).split()

def tokenize(line):
    return line.split()

def translate(model, data, device, bre, text):
    """Translate input text with trained model and output new text."""
    text_bpe = bpe_tokenize(text)
    SRC = data['vocab']['src']

    print(text_bpe)
    unk_idx = data['vocab']['src'].vocab.stoi[SRC.unk_token]  # data['tokens][unk_idx]
    src_seq = [data['vocab']['src'].vocab.stoi.get(word, unk_idx) for word in text_bpe]
    print(src_seq)
    for i in src_seq:
        try:
            print(data['vocab']['src'].vocab.itos[i], end=" ")
        except IndexError:
            print("unk", end=" ")
    pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
    print(pred_seq)
    # bpe decoding
    
    pred_line = ' '.join(data['vocab']['trg'].vocab.itos[idx] for idx in pred_seq)
    pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')
    return pred_line.strip() + '\n'

     
print(translate(translator, data, device, bpe, "Мало людей обладают большим энтузиазмом для ООН."))
     

if __name__ == "__main__":
    '''
    Usage: python translate.py -model trained.chkpt -data multi30k.pt -no_cuda
    '''
    main()
