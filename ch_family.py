from enums.enum_ch_family import Enum_Ch_Family


class ChFamilyMethods():
    digits = "1234567890"
    punctuations = '.,;:!?'

    @staticmethod
    def select_method(enum_Ch_Family):
        """

        :type enum_Ch_Family: Enum_Ch_Family
        """
        d = ChFamilyMethods.dict_family_method()
        if enum_Ch_Family in d:
            return d[enum_Ch_Family]

        return None

    @staticmethod
    def dict_family_method():
        return {
            Enum_Ch_Family.PREFIX: ChFamilyMethods.is_prefix,
            Enum_Ch_Family.SUFFIX: ChFamilyMethods.is_suffix,
            Enum_Ch_Family.SPACE_PREFIX: ChFamilyMethods.is_space_prefix,
            Enum_Ch_Family.SPACE_SUFFIX: ChFamilyMethods.is_space_suffix,
            Enum_Ch_Family.WHOLE_WORD: ChFamilyMethods.is_whole_word,
            Enum_Ch_Family.MID_WORD: ChFamilyMethods.is_mid_word,
            Enum_Ch_Family.MULTI_WORD: ChFamilyMethods.multi_word,
            Enum_Ch_Family.BEG_PUNCT: ChFamilyMethods.beg_punct,
            Enum_Ch_Family.MID_PUNCT: ChFamilyMethods.mid_punct,
            Enum_Ch_Family.END_PUNCT: ChFamilyMethods.end_punct,
            Enum_Ch_Family.N_GRAM_IN_TOKEN: ChFamilyMethods.is_n_gram_in_token
        }

    #########
    @staticmethod
    def is_prefix(target, preview=None, next=None):
        """Is a fragment of word (not a word), and the preview character is'nt.

        :type next: str
        :type preview: str
        :type target: str
        """

        return ChFamilyMethods.is_word_fragment(target) and ((preview is None) or not ChFamilyMethods.is_word_fragment(
            preview[0])) and (next is not None and ChFamilyMethods.is_word_fragment(next))

    @staticmethod
    def is_suffix(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return ChFamilyMethods.is_word_fragment(target) and preview is not None and ChFamilyMethods.is_word_fragment(
            preview[0]) and (
                   next is None or not ChFamilyMethods.is_word_fragment(next[-1]))

    @staticmethod
    def is_space_prefix(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return target.startswith(' ') and (
            len(target) == 1 or ChFamilyMethods.is_prefix(target[1:], preview=preview[1:] if preview else preview,
                                                          next=next[1:] if next else next))

    @staticmethod
    def is_space_suffix(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return target.endswith(' ') and (
            len(target) == 1 or ChFamilyMethods.is_suffix(target[:-1], preview=preview[:-1] if preview else preview,
                                                          next=next[:-1] if next else next))

    #########

    @staticmethod
    def is_whole_word(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return ChFamilyMethods.is_word_fragment(target) and not (
            ChFamilyMethods.is_suffix(target, preview, next) or ChFamilyMethods.is_prefix(target, preview,
                                                                                          next)) and not ChFamilyMethods.is_mid_word(
            target, preview, next)

    @staticmethod
    def is_mid_word(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return ChFamilyMethods.is_word_fragment(target) and not (
            ChFamilyMethods.is_suffix(target, preview, next) or ChFamilyMethods.is_prefix(target, preview,
                                                                                          next)) and preview is not None and ChFamilyMethods.is_word_fragment(
            preview[0]) and next is not None and ChFamilyMethods.is_word_fragment(next[-1])

    @staticmethod
    def multi_word(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        if not ' ' in target:
            return False

        for punc in ChFamilyMethods.punctuations:
            if punc in target:
                return False

        count = 0
        for w in target.split(' '):
            if not ChFamilyMethods.is_word_fragment(w):
                return False
            if len(w) > 0:
                count += 1

        return count > 1

    @staticmethod
    def beg_punct(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return target[0] in ChFamilyMethods.punctuations

    @staticmethod
    def mid_punct(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        for ch in target[1:-1]:
            if ch in ChFamilyMethods.punctuations:
                return True
        return False

    @staticmethod
    def end_punct(target, preview=None, next=None):
        """

        :type next: str
        :type preview: str
        :type target: str
        """
        return ChFamilyMethods.beg_punct(target[::-1])

    # auxiliary methods

    @staticmethod
    def is_word_fragment(fragment):
        for ch in ChFamilyMethods.punctuations + ' "/\\*()$@+&#\n'+ChFamilyMethods.digits:
            if ch in fragment.lower():
                return False
        return True

    @staticmethod
    def is_none_or_startswith_space(fragment):
        return fragment is None or fragment.startswith(' ')

    @staticmethod
    def is_n_gram_in_token(fragment, p=None, n=None):
        return ' ' not in fragment

    @staticmethod
    def has_number(token):
        for char in token:
            if char in ChFamilyMethods.digits:
                return True
        return False

    @staticmethod
    def is_letter(char):
        return char.lower() in "abcdefghijklmnopqrstuvwxyzáéèíïóúñ"
