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
vocab_path = os.path.join(os.path.dirname(__file__), '../../../model/vocab')
vocab_transform = torch.load(vocab_path)

# Tokenizers
token_transform = {}
token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()

# Special tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 5  # EOS is [SEP]
CLS_IDX = 4  # [CLS] for Nepali

# Device
device = torch.device('cpu')  # Use CPU for inference

# Model parameters (from training)
input_dim = len(vocab_transform["en"])
output_dim = len(vocab_transform["ne"])
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

# Load model
model_path = os.path.join(os.path.dirname(__file__), '../../../model/multiplicative_Seq2SeqTransformer.pt')
params, state = torch.load(model_path, map_location=device)

enc = Encoder(input_dim, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, atten_type, device)
dec = Decoder(output_dim, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, atten_type, device)
model = Seq2SeqTransformer(enc, dec, PAD_IDX, PAD_IDX, device)
model.load_state_dict(state)
model.eval()
model.to(device)

# Shared WordPiece detokenizer (aligned with notebook)
SPECIAL_TOKENS = {'<sos>', '<eos>', '<pad>', '[CLS]', '[SEP]', '[UNK]', '[MASK]'}
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
    src_indexes = [vocab_transform["en"].get_stoi().get(token, UNK_IDX) for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    return tokens, src_tensor, src_mask


def greedy_decode(sentence: str, max_len: int = 50) -> List[str]:
    _, src_tensor, src_mask = _prepare_src(sentence)
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    ne_itos = vocab_transform["ne"].get_itos()
    trg_indexes: List[int] = [CLS_IDX]
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
            candidate_tok = ne_itos[candidate]
            if candidate_tok in {'[CLS]', '<sos>', '<pad>'}:
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

        if ne_itos[chosen] in {'[SEP]', '<eos>'} and step >= 2:
            break
        if len(trg_indexes) >= 4 and len(set(trg_indexes[-3:])) == 1:
            trg_indexes.append(EOS_IDX)
            break

    return [ne_itos[i] for i in trg_indexes]


def translate_sentence(sentence, model, vocab_transform, device, max_len=50):
    model.eval()
    trg_tokens = greedy_decode(sentence, max_len)
    filtered = [token for token in trg_tokens if token not in SPECIAL_TOKENS]
    return detokenize_wordpiece(filtered)


def translate_sentence_debug(sentence, model, vocab_transform, device, max_len=50):
    model.eval()
    tokens = greedy_decode(sentence, max_len)
    return detokenize_wordpiece([t for t in tokens if t not in SPECIAL_TOKENS])
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
