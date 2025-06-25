from typing import TypedDict, Union

from scipy.sparse import coo_matrix, csr_matrix, lil_matrix, spmatrix
from sparse import COO, DOK


class EdgeData(TypedDict):
    weight: float


SparseMatrix = Union[spmatrix,
                     coo_matrix,
                     csr_matrix,
                     lil_matrix,
                     COO,
                     DOK]

