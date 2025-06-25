from typing import List


class ApplicationException(Exception):
    def __init__(self, message: str) -> None:
        super(ApplicationException, self).__init__()
        self.message = message


class UnknownEnzymeException(ApplicationException):
    """All sequences were excluded during filtering"""
    def __init__(self, target: str, similar: List[str]) -> None:
        super(UnknownEnzymeException, self).__init__(
            '{} is undefined, but its similar to: {}'.format(target, ', '.join(similar)))


class UnknownOrientationStateException(ApplicationException):
    """All sequences were excluded during filtering"""
    def __init__(self, ori: str) -> None:
        super(UnknownOrientationStateException, self).__init__('unknown orientation state [{}].'.format(ori))


class NoneAcceptedException(ApplicationException):
    """All sequences were excluded during filtering"""
    def __init__(self) -> None:
        super(NoneAcceptedException, self).__init__('all sequences were excluded')


class TooFewException(ApplicationException):
    """Method requires a minimum of nodes"""
    def __init__(self, min_seq: int, method: str) -> None:
        super(TooFewException, self).__init__('More than {} sequences are required to apply {}'.format(min_seq, method))


class NoRemainingClustersException(ApplicationException):
    def __init__(self, msg: str) -> None:
        super(NoRemainingClustersException, self).__init__(msg)


class ReportFormatException(ApplicationException):
    """Clustering does not contain a report"""
    def __init__(self, context: str, _id: int) -> None:
        super(ReportFormatException, self).__init__(
            f'Report did not follow expected format for cluster {_id}: {context}')


class ZeroLengthException(ApplicationException):
    """Sequence of zero length"""
    def __init__(self, seq_name: str) -> None:
        super(ZeroLengthException, self).__init__('Sequence [{}] has zero length'.format(seq_name))


class NoRecordsException(ApplicationException):
    """No records were found"""
    def __init__(self, file_format: str, file_name: str) -> None:
        super(NoRecordsException, self).__init__(f'No {file_format} format records were found in {file_name}')

class NotFoundException(ApplicationException):
    """General Not Found exception"""
    def __init__(self, context: str, _id: str) -> None:
        super(NotFoundException, self).__init__(f'{context}:{_id} was not found')


class ParsingError(ApplicationException):
    """An error during input parsing"""
    def __init__(self, msg: str) -> None:
        super(ParsingError, self).__init__(msg)


class InvalidCoverageFormatError(ApplicationException):
    def __init__(self, seq_name: str, caller_name: str, txt: str) -> None:
        super(ApplicationException, self).__init__(
            f'Failed to extract coverage for {seq_name}. "{txt}" did not match {caller_name} pattern')


class RejectedSequenceException(ApplicationException):
    """Sequence failed some type of acceptance criteria"""
    def __init__(self) -> None:
        super(RejectedSequenceException, self).__init__('sequence has been rejected')
