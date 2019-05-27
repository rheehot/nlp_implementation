import json
import fire
import torch
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from model.utils import collate_fn
from model.data import Corpus
from model.net import SAN
from mecab import MeCab
from tqdm import tqdm


def get_accuracy(model, dataloader, device):
    if model.training:
        model.eval()

    correct_count = 0

    for mb in tqdm(dataloader, desc='steps'):
        queries_a_mb, queries_b_mb, y_mb = map(lambda elm: elm.to(device), mb)
        queries_mb = (queries_a_mb, queries_b_mb)

        with torch.no_grad():
            score, _, _ = model(queries_mb)
            y_mb_hat = torch.max(score, 1)[1]
            correct_count += (y_mb_hat == y_mb).sum().item()
    else:
        acc = correct_count / len(dataloader.dataset)
    return acc


def main(cfgpath):
    # parsing json
    proj_dir = Path.cwd()
    with open(proj_dir / cfgpath) as io:
        params = json.loads(io.read())

    # restoring model
    savepath = proj_dir / params['filepath'].get('ckpt')
    ckpt = torch.load(savepath)

    ## common params
    vocab_filepath = params['filepath'].get('vocab')
    with open(vocab_filepath, mode='rb') as io:
        vocab = pickle.load(io)

    num_classes = params['model'].get('num_classes')
    lstm_hidden_dim = params['model'].get('lstm_hidden_dim')
    hidden_dim = params['model'].get('hidden_dim')
    da = params['model'].get('da')
    r = params['model'].get('r')

    model = SAN(num_classes=num_classes, lstm_hidden_dim=lstm_hidden_dim, hidden_dim=hidden_dim,
                da=da, r=r, vocab=vocab)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # creating dataset, dataloader
    tokenizer = MeCab().morphs
    tr_filepath = proj_dir / params['filepath'].get('tr')
    val_filepath = proj_dir / params['filepath'].get('val')
    batch_size = params['training'].get('batch_size')

    tr_ds = Corpus(tr_filepath, tokenizer, vocab)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    val_ds = Corpus(val_filepath, tokenizer, vocab)
    val_dl = DataLoader(val_ds, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    tr_acc = get_accuracy(model, tr_dl, device)
    val_acc = get_accuracy(model, val_dl, device)

    print('tr_acc: {:.2%}, val_acc: {:.2%}'.format(tr_acc, val_acc))


if __name__ == '__main__':
    fire.Fire(main)