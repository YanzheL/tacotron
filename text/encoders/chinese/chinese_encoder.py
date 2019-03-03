from text.encoders import TextEncoder
from pypinyin import Style, pinyin
import re
from zhon import hanzi
import numpy as np


def _list_lookup(l, index, default=None):
    return l[index] if -len(l) <= index < len(l) else default


def _symbol_lookup(s, default=-1):
    found = np.where(SYMBOLS == s)[0]
    if len(found):
        return found[0]
    else:
        return default


def encode_pinyin(initial, final, tone):
    initial_idx = _symbol_lookup(initial)
    final_idx = _symbol_lookup(final)
    assert 0 <= tone <= 4, 'Invalid tone'
    return initial_idx, final_idx, tone


# Standard pinyin initials and finals from http://www.moe.gov.cn/ewebeditor/uploadfile/2015/03/02/20150302165814246.pdf

INITIALS = (
    'b', 'p', 'm', 'f',
    'd', 't', 'n', 'l',
    'g', 'k', 'h',
    'j', 'q', 'x',
    'zh', 'ch', 'sh', 'r',
    'z', 'c', 's'
)
FINALS = (
    'a',
    'o',
    'e',
    'ai',
    'ei',
    'ao',
    'ou',
    'an',
    'en',
    'ang',
    'eng',
    'ong',
    'i',
    'ia',
    'ie',
    'iao',
    'iou',
    'ian',
    'in',
    'iang',
    'ing',
    'iong',
    'u',
    'ua',
    'uo',
    'uai',
    'uei',
    'uan',
    'uen',
    'uang',
    'ueng',
    'v',
    've',
    'van',
    'vn'
)

_STOP_GENERAL = '{STOP}'
_NON_STOP_GENERAL = '{NON_STOP}'
_CH_DIVIDER = '{DIV}'
_EOS = '{EOS}'

SYMBOLS = np.array(INITIALS + FINALS + (_STOP_GENERAL, _NON_STOP_GENERAL, _CH_DIVIDER, _EOS))
CODE_STOP_GENERAL = _symbol_lookup(_STOP_GENERAL)
CODE_NON_STOP_GENERAL = _symbol_lookup(_NON_STOP_GENERAL)
CODE_CH_DIVIDER = _symbol_lookup(_CH_DIVIDER)
CODE_EOS = _symbol_lookup(_EOS)

_tone_re = re.compile(r'(\d)')
_characters_re = re.compile('[{}]'.format(hanzi.characters))
_stops_re = re.compile('[{}]'.format(hanzi.stops))
_non_stops_re = re.compile('[{}]'.format(hanzi.non_stops))


def _match_indices(pat, s):
    return [m.start(0) for m in pat.finditer(s)]


def _encode_characters(text):
    pinyins = []
    initials = pinyin(text, Style.INITIALS, strict=True)
    finals = pinyin(text, Style.FINALS, strict=True)
    tones = pinyin(text, Style.TONE3, strict=True)
    for initial_l, final_l, tone_l in zip(initials, finals, tones):
        initial = _list_lookup(initial_l, 0, '')
        # final and tone must exist
        final = final_l[0]
        tone_s = tone_l[0]

        tone_found = _tone_re.findall(tone_s)
        tone_len = len(tone_found)
        if tone_len == 0:
            tone = 0
        elif tone_len == 1:
            tone = int(tone_found[0])
        else:
            raise ValueError('Multiple tones found, maybe a bug')
        py = encode_pinyin(initial, final, tone)
        pinyins.append(py)
    return pinyins


class ChineseEncoder(TextEncoder):
    LANGUAGE = 'chinese'
    SYMBOLS_SIZE = len(SYMBOLS)

    @staticmethod
    def encode(text):
        text_array = np.array(list(text))
        characters_idx = _match_indices(_characters_re, text)
        stops_idx = _match_indices(_stops_re, text)
        non_stops_idx = _match_indices(_non_stops_re, text)
        pinyin_pairs = _encode_characters(text_array[characters_idx])
        encoded = []
        ch = 0
        for pos in range(len(text)):
            if pos in stops_idx:
                encoded.append(CODE_STOP_GENERAL)
            elif pos in non_stops_idx:
                encoded.append(CODE_NON_STOP_GENERAL)
            elif pos in characters_idx:
                encoded.extend(pinyin_pairs[ch])
                encoded.append(CODE_CH_DIVIDER)
                ch += 1
        encoded.append(CODE_EOS)
        encoded = np.array(encoded, np.int32)
        return np.extract(encoded >= 0, encoded)


__all__ = ['ChineseEncoder']

if __name__ == '__main__':
    # print(INITIAL_BITS)
    encoder = TextEncoder.get_encoder('chinese')

    # encode = encoder.encode('当时就是不遵守上面的规则来处理声母和韵母，比如：会被当做声母，(迂) 的韵母就是一般认为的等。')
    encode = encoder.encode('我我我我')
    print(encode)

    pass
