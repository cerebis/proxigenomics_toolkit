
class ApplicationException(Exception):
    def __init__(self, message):
        super(ApplicationException, self).__init__(message)


class UnknownEnzymeException(ApplicationException):
    """All sequences were excluded during filtering"""
    def __init__(self, target, similar):
        super(UnknownEnzymeException, self).__init__(
            '{} is undefined, but its similar to: {}'.format(target, ', '.join(similar)))


class UnknownOrientationStateException(ApplicationException):
    """All sequences were excluded during filtering"""
    def __init__(self, ori):
        super(UnknownOrientationStateException, self).__init__('unknown orientation state [{}].'.format(ori))


class NoneAcceptedException(ApplicationException):
    """All sequences were excluded during filtering"""
    def __init__(self):
        super(NoneAcceptedException, self).__init__('all sequences were excluded')


class TooFewException(ApplicationException):
    """Method requires a minimum of nodes"""
    def __init__(self, minseq, method):
        super(TooFewException, self).__init__('More than {} sequences are required to apply {}'.format(minseq, method))


class NoRemainingClustersException(ApplicationException):
    def __init__(self, msg):
        super(NoRemainingClustersException, self).__init__(msg)


class ReportFormatException(ApplicationException):
    """Clustering does not contain a report"""
    def __init__(self, context, _id):
        super(ReportFormatException, self).__init__(
            f'Report did not follow expected format for cluster {_id}: {context}')


class ZeroLengthException(ApplicationException):
    """Sequence of zero length"""
    def __init__(self, seq_name):
        super(ZeroLengthException, self).__init__('Sequence [{}] has zero length'.format(seq_name))


class NotFoundException(ApplicationException):
    """General Not Found exception"""
    def __init__(self, context, _id):
        super(NotFoundException, self).__init__(f'{context}:{_id} was not found')


class ParsingError(ApplicationException):
    """An error during input parsing"""
    def __init__(self, msg):
        super(ParsingError, self).__init__(msg)


class InvalidCoverageFormatError(ApplicationException):
    def __init__(self, seq_name, caller_name, txt):
        super(ApplicationException, self).__init__(
            f'Failed to extract coverage for {seq_name}. "{txt}" did not match {caller_name} pattern')


class RejectedSequenceException(ApplicationException):
    """Sequence failed some type of acceptance criteria"""
    def __init__(self):
        super(RejectedSequenceException, self).__init__('sequence has been rejected')
