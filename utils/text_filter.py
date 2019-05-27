import re


def __getRoNumberPlateRegex():
    """
    Get specific romanian number plate regex.
    """
    # specific romanian number plate regex creation
    nr_regex = "B\\d{2,3}[A-Z]{3}"
    county_abbreviation = ["AB", "AG", "AR", "BC", "BH", "BN", "BR", "BT", "BV", "BZ", "CJ", "CL", "CS", "CT", "CV",
                           "DB", "DJ", "GJ", "GL", "GR", "HD", "HR", "IF", "IL", "IS", "MH", "MM", "MS", "NT", "OT",
                           "PH", "SB", "SJ", "SM", "TL", "TM", "TR", "VL", "VN", "VS"]

    for county in county_abbreviation:
        nr_regex += "|" + county + "\\d{2}[A-Z]{3}"
    return nr_regex


def _filterRomanianNumbers(number_candidates):
    """
    Filter the romanian number plates from the potential candidates.
    :param number_candidates:
    :return: the filtered romanian number plates
    """
    numbers = []

    for number in number_candidates:
        # workarounds
        if number[0:2] == "CI":
            number = str(number).replace(number[0:2], "CJ", 1)

        nr_regex = __getRoNumberPlateRegex()

        ro_nr_pattern = re.compile(nr_regex)
        ro_nrs = re.findall(ro_nr_pattern, number)
        if ro_nrs:
            numbers.append(ro_nrs[0])

    return numbers


class NprTextsFilter:
    """
    Class that filters out the dates and the number plate texts from a list of potential candidates.
    """
    __date_pattern = None
    __number_pattern = None

    def __init__(self) -> None:
        self.__date_pattern = re.compile("^\\d{1,2}/\\d{1,2}/\\d{4}$|^\\d{4}/\\d{1,2}/\\d{2}$")
        self.__number_pattern = re.compile("[A-Z]{2}\\d{2}[A-Z]{3}|[B]\\d{2,3}[A-Z]{3}")

    def filterDatesAndPlates(self, strings):
        """
        Detect the dates and Romanian number plates texts from all detected text.

        :param strings: the list of texts from the detection
        :return: tuple of 2 lists: (dates, numbers)
        """

        dates = []
        number_candidates = []

        for text in strings:
            # filter out not needed spaces from within the text
            text = text.replace(" ", "")

            # find the date from within the text
            found_dates = re.findall(self.__date_pattern, text)
            if found_dates:
                dates.append(found_dates[0])

            # find the potential number from within the text
            # further filtering will be done
            found_numbers = re.findall(self.__number_pattern, text)
            if found_numbers:
                number_candidates.append(found_numbers[0])

        # filtering with a specific romanian number plate regex
        numbers = _filterRomanianNumbers(number_candidates)

        return dates, numbers
