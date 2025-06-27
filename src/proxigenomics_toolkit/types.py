from collections import namedtuple
from typing import Optional, TypedDict, Union

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, spmatrix
from sparse import COO, DOK


class EdgeData(TypedDict):
    """
    A basic weighted edge data type.
    :ivar weight: The weight of the edge.
    :type weight: float
    """
    weight: float


SparseMatrix = Union[spmatrix,
coo_matrix,
csr_matrix,
lil_matrix,
COO,
DOK]

# Contact map type definitions

# TODO remove uses of this namedtuple in preference for the equivalent Numpy structured array
SeqInfo = namedtuple('SeqInfo',
                     ['offset', 'refid', 'name', 'length', 'sites', 'gc'])

SEQ_INFO_NPTYPE = np.dtype([('offset', int),
                            ('refid', int),
                            ('name', 'U30'),
                            ('length', int),
                            ('sites', int),
                            ('gc', float)])

# Ordering type definitions

STRUCT_NPTYPE = np.dtype([('pos', np.int32),
                          ('ori', np.int8),
                          ('mask', bool),
                          ('length', np.int32)])

INDEX_NPTYPE = np.dtype([('index', np.int32),
                         ('ori', np.int8)])

# Clustering type defintions

FULL_REPORT_NPTYPE = np.dtype([('length', np.int64),
                               ('gc', np.float64),
                               ('cov', np.float64)])

MINIMAL_REPORT_NPTYPE = np.dtype([('length', np.int64),
                                  ('gc', np.float64)])

ReportArray = Union[npt.NDArray[FULL_REPORT_NPTYPE],
npt.NDArray[MINIMAL_REPORT_NPTYPE]]


class ClusterType(TypedDict):
    """
    Represents a data structure to define properties of a cluster in a typed
    and structured format. Primarily used to store and manage metadata and
    attributes associated with a cluster object in computational or analytical
    workflows.

    :ivar name: The name of the cluster.
    :type name: str
    :ivar seq_names: An array of sequence names associated with the cluster.
    :type seq_names: npt.NDArray[str]
    :ivar seq_ids: An array of internal sequence identifiers linked to the cluster.
    :type seq_ids: npt.NDArray[int]
    :ivar extent: The extent or total length of the cluster in nucleotides.
    :type extent: int
    :ivar status: The current status or state of the cluster.
    :type status: str
    :ivar unreferenced: An optional array of unreferenced sequences in the
        cluster.
    :type unreferenced: Optional[npt.NDArray[str]]
    :ivar deduplicated: An optional array of deduplicated sequences in the
        cluster.
    :type deduplicated: Optional[npt.NDArray[str]]
    :ivar report: An optional structured report detailing characteristics of
    sequences associated with the cluster.
    :type report: Optional[ReportArray]
    """
    name: str
    seq_names: npt.NDArray[str]
    seq_ids: npt.NDArray[int]
    extent: int
    status: str
    unreferenced: Optional[npt.NDArray[str]]
    deduplicated: Optional[npt.NDArray[str]]
    report: Optional[ReportArray]
