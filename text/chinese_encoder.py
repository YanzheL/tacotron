from text.textencoder import TextEncoder
from pypinyin import Style, pinyin
import re
from zhon import hanzi
import numpy as np


class ChineseEncoder(TextEncoder):
    '''
    -----------------------------------
    |  INITIAL  |    FINAL    | TONE  |
    | _ _ _ _ _ | _ _ _ _ _ _ | _ _ _ |
    '''
    LANGUAGE = 'chinese'
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
        'ong'
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

    NON_STOPS = hanzi.non_stops
    STOPS = hanzi.stops

    INITIAL_BITS = 5
    FINAL_BITS = 6
    TONE_BITS = 3

    _tone_re = re.compile(r'(\d)')
    _characters_re = re.compile('[{}]'.format(hanzi.characters))
    _stops_re = re.compile('[{}]'.format(hanzi.stops))
    _non_stops_re = re.compile('[{}]'.format(hanzi.non_stops))

    @staticmethod
    def mask(bits):
        return ~ (-1 >> bits << bits)

    @staticmethod
    def match_indices(pat, s):
        return [m.start(0) for m in pat.finditer(s)]

    @staticmethod
    def encode(text):
        text_array = np.array(list(text))
        characters_idx = ChineseEncoder.match_indices(ChineseEncoder._characters_re, text)
        stops_idx = ChineseEncoder.match_indices(ChineseEncoder._stops_re, text)
        non_stops_idx = ChineseEncoder.match_indices(ChineseEncoder._non_stops_re, text)
        print('characters_idx={}, stops_idx={}, non_stops_idx={}'.format(
            characters_idx, stops_idx, non_stops_idx
        )

    @staticmethod
    def _encode_non_characters(text):
        pass

    @staticmethod
    def _encode_characters(text):
        sequences = []
        initials = pinyin(text, Style.INITIALS, strict=True)
        finals = pinyin(text, Style.FINALS, strict=True)
        tones = pinyin(text, Style.TONE3, strict=True)
        for initial_l, final_l, tone_l in zip(initials, finals, tones):
            initial = ChineseEncoder._get_from_list(initial_l, 0, '')
            # final and tone must exist
            final = final_l[0]
            tone_s = tone_l[0]

            tone_found = ChineseEncoder._tone_re.findall(tone_s)
            tone_len = len(tone_found)
            if tone_len == 0:
                tone = 0
            elif tone_len == 1:
                tone = int(tone_found[0])
            else:
                raise ValueError('Multiple tones found, maybe a bug')
            code = ChineseEncoder._encode_pinyin(initial, final, tone)
            sequences.append(code)
        return sequences

    @staticmethod
    def _get_from_list(l, index, default=None):
        return l[index] if -len(l) <= index < len(l) else default

    @staticmethod
    def _get_from_tuple(t, value, default=None):
        try:
            idx = t.index(value)
        except ValueError:
            idx = default
        return idx

    @staticmethod
    def _encode_pinyin(initial, final, tone):
        initial_idx = ChineseEncoder._get_from_tuple(ChineseEncoder.INITIALS, initial, -1)
        final_idx = ChineseEncoder._get_from_tuple(ChineseEncoder.FINALS, final, -1)
        assert 0 <= tone <= 4, 'Invalid tone'

        initial_idx &= ChineseEncoder.mask(ChineseEncoder.INITIAL_BITS)
        final_idx &= ChineseEncoder.mask(ChineseEncoder.FINAL_BITS)
        code = tone
        shift = ChineseEncoder.TONE_BITS
        code += final_idx << shift
        shift += ChineseEncoder.FINAL_BITS
        code += initial_idx << shift
        print('initial=({},{}), final=({},{}), tone=({}), code=({})'.format(
            initial, initial_idx, final, final_idx, tone, bin(code))
        )
        return code

    @staticmethod
    def round_up_by_2(v):
        v -= 1
        v |= v >> 1
        v |= v >> 2
        v |= v >> 4
        v |= v >> 8
        v |= v >> 16
        v += 1
        return v


if __name__ == '__main__':
    encoder = TextEncoder.get_encoder('chinese')

    print(encoder.encode('当时就是不遵守上面的规则来处理声母和韵母，比如：会被当做声母，(迂) 的韵母就是一般认为的等。'))

    pass
