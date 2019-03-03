from text import TextEncoder
from pypinyin import Style, pinyin
import re
from zhon import hanzi
import numpy as np


class ChineseEncoder(TextEncoder):
    '''
    ----------------------------------------
    | SYM |  INITIAL  |    FINAL    | TONE  |
    | _ _ | _ _ _ _ _ | _ _ _ _ _ _ | _ _ _ |

    '''
    LANGUAGE = 'chinese'

    @staticmethod
    def encode(text):
        text_array = np.array(list(text))
        encoded = np.zeros(text_array.shape, dtype=np.int32)
        characters_idx = match_indices(_characters_re, text)
        stops_idx = match_indices(_stops_re, text)
        non_stops_idx = match_indices(_non_stops_re, text)
        # print('text_array={}\ncharacters_idx={}\nstops_idx={}\nnon_stops_idx={}'.format(
        #     text_array, characters_idx, stops_idx, non_stops_idx
        # ))
        characters_encode = _encode_characters(text_array[characters_idx])
        # stops_encode = _encode_stops(text_array[stops_idx])
        # non_stops_encode = _encode_non_stops(text_array[non_stops_idx])

        encoded[characters_idx] = characters_encode
        encoded[stops_idx] = STOPS_ENCODE
        encoded[non_stops_idx] = NON_STOP_ENCODE

        #         print(
        #             '''
        # text={}
        # text_array={}
        # characters_idx={}
        # characters_encode={}
        # stops_idx={}
        # stops_encode={}
        # non_stops_idx={}
        # non_stops_encode={}
        # encode={}
        #             '''.format(
        #                 text,
        #                 text_array,
        #                 characters_idx,
        #                 [bin(i) for i in characters_encode],
        #                 stops_idx,
        #                 bin(STOPS_ENCODE),
        #                 non_stops_idx,
        #                 bin(NON_STOP_ENCODE),
        #                 [bin(i) for i in encoded]
        #             )
        #         )

        return encoded


def mask(bits):
    return ~ (-1 >> bits << bits)


def match_indices(pat, s):
    return [m.start(0) for m in pat.finditer(s)]


def _encode_non_stops(text):
    pass


def _encode_stops(text):
    pass


def _encode_characters(text):
    sequences = []
    initials = pinyin(text, Style.INITIALS, strict=True)
    finals = pinyin(text, Style.FINALS, strict=True)
    tones = pinyin(text, Style.TONE3, strict=True)
    for initial_l, final_l, tone_l in zip(initials, finals, tones):
        initial = _get_from_list(initial_l, 0, '')
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
        code = _encode_pinyin(initial, final, tone)
        sequences.append(code)
    return sequences


def _get_from_list(l, index, default=None):
    return l[index] if -len(l) <= index < len(l) else default


def _get_from_tuple(t, value, default=None):
    try:
        idx = t.index(value)
    except ValueError:
        idx = default
    return idx


def _encode_pinyin(initial, final, tone):
    initial_idx = _get_from_tuple(INITIALS, initial, -1)
    final_idx = _get_from_tuple(FINALS, final, -1)
    assert 0 <= tone <= 4, 'Invalid tone'

    code = concat_binarys(
        [
            tone, final_idx, initial_idx
        ],
        [
            TONE_BITS, FINAL_BITS, INITIAL_BITS
        ]
    )
    print('initial=({},{}), final=({},{}), tone=({}), code=({})'.format(
        initial, initial_idx, final, final_idx, tone, bin(code))
    )
    return code


def concat_binarys(codes, bits):
    encode = 0
    shift = 0
    for code, bit in zip(codes, bits):
        tp = code & mask(bit)
        tp <<= shift
        encode += tp
        shift += bit
    return encode


def round_up_by_2(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


# All standard initials and finals from http://www.moe.gov.cn/ewebeditor/uploadfile/2015/03/02/20150302165814246.pdf

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

NON_STOPS = tuple(hanzi.non_stops)
STOPS = tuple(hanzi.stops)

INITIAL_BITS = int(np.log2(round_up_by_2(len(INITIALS))))
FINAL_BITS = int(np.log2(round_up_by_2(len(FINALS))))
TONE_BITS = 3
SYMBOL_BITS = 2

STOPS_ENCODE = 1 << INITIAL_BITS + FINAL_BITS + TONE_BITS
NON_STOP_ENCODE = 2 << INITIAL_BITS + FINAL_BITS + TONE_BITS

_tone_re = re.compile(r'(\d)')
_characters_re = re.compile('[{}]'.format(hanzi.characters))
_stops_re = re.compile('[{}]'.format(hanzi.stops))
_non_stops_re = re.compile('[{}]'.format(hanzi.non_stops))

__all__ = ['ChineseEncoder']

if __name__ == '__main__':
    # print(INITIAL_BITS)
    encoder = TextEncoder.get_encoder('chinese')

    encode = encoder.encode('当时就是不遵守上面的规则来处理声母和韵母，比如：会被当做声母，(迂) 的韵母就是一般认为的等。')

    pass