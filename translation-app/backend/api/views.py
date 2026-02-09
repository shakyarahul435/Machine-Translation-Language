import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import (
    Seq2SeqTransformer, Encoder, Decoder, EncoderLayer, DecoderLayer,
    MultiHeadAttentionLayer, PositionwiseFeedforwardLayer, AdditiveAttention
)
import os
import sys
import re
from typing import List

# Fix for pickle loading - add all model classes to __main__
sys.modules['__main__'].Encoder = Encoder
sys.modules['__main__'].Decoder = Decoder
sys.modules['__main__'].EncoderLayer = EncoderLayer
sys.modules['__main__'].DecoderLayer = DecoderLayer
sys.modules['__main__'].MultiHeadAttentionLayer = MultiHeadAttentionLayer
sys.modules['__main__'].PositionwiseFeedforwardLayer = PositionwiseFeedforwardLayer
sys.modules['__main__'].AdditiveAttention = AdditiveAttention
sys.modules['__main__'].Seq2SeqTransformer = Seq2SeqTransformer

# Load vocab
vocab_path = os.path.join(os.path.dirname(__file__), '../../../model/vocab_latest.pt')
vocab_transform = torch.load(vocab_path)

src_vocab = vocab_transform["en"]
trg_vocab = vocab_transform["ne"]
src_stoi = src_vocab.get_stoi()
trg_stoi = trg_vocab.get_stoi()
src_itos = src_vocab.get_itos()
trg_itos = trg_vocab.get_itos()

# Tokenizers
token_transform = {}
token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()

# Special tokens
UNK_IDX = src_stoi.get('<unk>', 0)
SRC_PAD_IDX = src_stoi.get('<pad>', 1)
TRG_PAD_IDX = trg_stoi.get('<pad>', 1)
SOS_IDX = trg_stoi.get('<sos>', 2)
EOS_IDX = trg_stoi.get('<eos>', 3)

# Device
device = torch.device('cpu')  # Use CPU for inference

# Model parameters (from training)
input_dim = len(src_vocab)
output_dim = len(trg_vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
atten_type = "multiplicative"  # Best model
MAX_SEQ_LEN = 512

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../../../model/multiplicative_seq2seq_lr1e-4_ep20.pt')
checkpoint = torch.load(model_path, map_location=device)

state = None
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state = checkpoint['state_dict']
    hyperparams = checkpoint.get('hyperparams', {})
    HID_DIM = hyperparams.get('hid_dim', HID_DIM)
    ENC_LAYERS = hyperparams.get('enc_layers', ENC_LAYERS)
    DEC_LAYERS = hyperparams.get('dec_layers', DEC_LAYERS)
    ENC_HEADS = hyperparams.get('enc_heads', ENC_HEADS)
    DEC_HEADS = hyperparams.get('dec_heads', DEC_HEADS)
    ENC_PF_DIM = hyperparams.get('enc_pf_dim', ENC_PF_DIM)
    DEC_PF_DIM = hyperparams.get('dec_pf_dim', DEC_PF_DIM)
    ENC_DROPOUT = hyperparams.get('enc_dropout', ENC_DROPOUT)
    DEC_DROPOUT = hyperparams.get('dec_dropout', DEC_DROPOUT)
    atten_type = hyperparams.get('atten_type', atten_type)
    input_dim = hyperparams.get('input_dim', input_dim)
    output_dim = hyperparams.get('output_dim', output_dim)
else:
    params, state = checkpoint
    HID_DIM = params.get('hid_dim', HID_DIM)
    ENC_LAYERS = params.get('enc_layers', ENC_LAYERS)
    DEC_LAYERS = params.get('dec_layers', DEC_LAYERS)
    ENC_HEADS = params.get('enc_heads', ENC_HEADS)
    DEC_HEADS = params.get('dec_heads', DEC_HEADS)
    ENC_PF_DIM = params.get('enc_pf_dim', ENC_PF_DIM)
    DEC_PF_DIM = params.get('dec_pf_dim', DEC_PF_DIM)
    ENC_DROPOUT = params.get('enc_dropout', ENC_DROPOUT)
    DEC_DROPOUT = params.get('dec_dropout', DEC_DROPOUT)
    atten_type = params.get('atten_type', atten_type)
    input_dim = params.get('input_dim', input_dim)
    output_dim = params.get('output_dim', output_dim)

enc = Encoder(input_dim, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, atten_type, device, max_length=MAX_SEQ_LEN)
dec = Decoder(output_dim, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, atten_type, device, max_length=MAX_SEQ_LEN)
model = Seq2SeqTransformer(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device)
model.load_state_dict(state)
model.eval()
model.to(device)

# Shared WordPiece detokenizer (aligned with notebook)
SPECIAL_TOKENS = {'<sos>', '<eos>', '<pad>', '<unk>', '[CLS]', '[SEP]', '[UNK]', '[MASK]'}
PUNCT_NO_SPACE_BEFORE = {',', '.', ':', ';', ')', ']', '}', '।', '!', '?'}
PUNCT_NO_SPACE_AFTER  = {'(', '[', '{'}

def detokenize_wordpiece(tokens: List[str]) -> str:
    words: List[str] = []
    for tok in tokens:
        if tok in SPECIAL_TOKENS:
            continue
        if tok.startswith('##'):
            sub = tok[2:]
            if words:
                words[-1] = words[-1] + sub
            else:
                words.append(sub)
        else:
            if tok in PUNCT_NO_SPACE_BEFORE and words:
                words[-1] = words[-1] + tok
            elif tok in PUNCT_NO_SPACE_AFTER:
                words.append(tok)
            else:
                words.append(tok)
    text = ' '.join(words)
    text = text.replace('( ', '(').replace(' )', ')').replace('[ ', '[').replace(' ]', ']')
    text = re.sub(r'([,.:;\)\]\}।!?])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _prepare_src(sentence: str):
    if callable(token_transform["en"]):
        tokens = token_transform["en"](sentence.lower())
    else:
        tokens = token_transform["en"].encode(sentence.lower()).tokens
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = [src_stoi.get(token, UNK_IDX) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    return tokens, src_tensor, src_mask


def greedy_decode(sentence: str, max_len: int = 50) -> List[str]:
    _, src_tensor, src_mask = _prepare_src(sentence)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes: List[int] = [SOS_IDX]
    last_id = None

    for step in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            logits, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        next_logits = logits[:, -1, :]
        topk = torch.topk(next_logits, k=min(5, next_logits.shape[-1]), dim=-1).indices.squeeze(0).tolist()

        chosen = None
        for candidate in topk:
            candidate_tok = trg_itos[candidate]
            if candidate_tok in {'<sos>', '<pad>', '[CLS]'}:
                continue
            if step < 2 and (candidate_tok in {'[SEP]', '<unk>', '[UNK]', '<eos>'} or candidate == EOS_IDX):
                continue
            if last_id is not None and candidate == last_id:
                continue
            chosen = candidate
            break
        if chosen is None:
            chosen = topk[0]

        trg_indexes.append(chosen)
        last_id = chosen

        if trg_itos[chosen] in {'[SEP]', '<eos>'} and step >= 2:
            break
        if len(trg_indexes) >= 4 and len(set(trg_indexes[-3:])) == 1:
            trg_indexes.append(EOS_IDX)
            break

    return [trg_itos[i] for i in trg_indexes]


def translate_sentence(sentence, model, vocab_transform, device, max_len=50):
    model.eval()
    trg_tokens = greedy_decode(sentence, max_len)
    filtered = [token for token in trg_tokens if token not in SPECIAL_TOKENS]
    return detokenize_wordpiece(filtered)


def translate_sentence_debug(sentence, model, vocab_transform, device, max_len=50):
    model.eval()
    tokens = greedy_decode(sentence, max_len)
    return [t for t in tokens if t not in SPECIAL_TOKENS]
def api_root(request):
    return Response({
        "message": "Translation API",
        "endpoints": {
            "translate": "/api/translate/ (POST with {'text': 'your text'})"
        }
    })

@api_view(['POST'])
def translate(request):
    try:
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'No text provided'}, status=400)

        translation = translate_sentence(text, model, vocab_transform, device)
        debug_tokens = translate_sentence_debug(text, model, vocab_transform, device)
        # Also get input tokens
        if callable(token_transform["en"]):
            input_tokens = token_transform["en"](text.lower())
        else:
            input_tokens = token_transform["en"].encode(text.lower()).tokens
        input_tokens = ['<sos>'] + input_tokens + ['<eos>']
        return Response({
            'translation': translation,
            'debug_tokens': debug_tokens,
            'input_tokens': input_tokens
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)
