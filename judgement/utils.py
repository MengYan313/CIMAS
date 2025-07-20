import pandas as pd
from difflib import SequenceMatcher
import re

class Utils:
    @staticmethod
    def compare_strings(str1, str2):
        # Remove all format characters and spaces from the string
        cleaned_str1 = ''.join(str1.split())
        cleaned_str2 = ''.join(str2.split())
        # Compare whether the processed strings are the same
        return cleaned_str1 == cleaned_str2

    @staticmethod
    def compare_contain_strings(str1, str2):
        # Remove all format characters and spaces from the string
        cleaned_str1 = ''.join(str1.split())
        cleaned_str2 = ''.join(str2.split())
        # Determine that str1 is not in str2 and str2 is not in str1
        return cleaned_str1 not in cleaned_str2 and cleaned_str2 not in cleaned_str1

    @staticmethod
    def contains_string(string_list, target_str):
        # Determine whether the list contains a specific string
        return any(Utils.compare_strings(target_str, s) for s in string_list)

    @staticmethod
    def merge_lists(list1, list2):
        # Check whether there are duplicate elements in the two lists
        if len(set(list1) & set(list2)) > 0:
            return [-1]
        else:  # Merge two lists and return the sorted list
            return sorted(list1 + list2)

    @staticmethod
    def code_filter(s: str) -> bool:
        parts = s.split() # Split by whitespace by default
        if not parts or parts[0] == "return": # Empty list or the first element is return
            return True
        if len(parts) == 3 and parts[1] in {"==", "!=", ">", ">=", "<", "<=", "&&", "||"}: # Three elements and the second element is a comparison operator
            return True
        if len(parts) == 3 and parts[1] == "=": # Simple assignment expression
            symbols = {"(", ")", ".", ";", "->"}
            if not any(symbol in s for symbol in symbols): # If none of the parentheses, dot, or semicolon are present
                return True
        return False

    @staticmethod
    def has_var_const_pattern(s: str) -> bool:
        """
        Determine whether there is a substring in the string in the form of <...>
        """
        pattern = r"<[^>]+>"
        return re.search(pattern, s) is not None

def main():
    code1 = "return MajorType::ByteString"
    code2 = "m_willBeWrittenTo = true"
    code3 = "p && *p"
    code10 = "(_flags & NEEDS_NEWLINE_NORMALIZATION) && *p == CR"
    code11 = "node = node->_next"
    codes = [code1, code2, code3, code10, code11]
    for code in codes:
        print(f"Code: {code}, Filtered: {Utils.code_filter(code)}")

if __name__ == "__main__":
    main()