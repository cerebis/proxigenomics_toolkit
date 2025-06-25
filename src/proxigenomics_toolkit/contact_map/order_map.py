import logging
import os
from typing import Optional

import numpy as np

from ..exceptions import NoneAcceptedException, TooFewException
from . import ContactMap

logger = logging.getLogger(__name__)


def order_clusters(contact_map: ContactMap,
                   clustering: dict,
                   seed: int,
                   cl_list: Optional[list]=None,
                   min_len: Optional[int]=None,
                   min_sig: Optional[int]=None,
                   max_fold: Optional[float]=None,
                   min_extent: Optional[int]=None,
                   min_size: Optional[int]=1,
                   work_dir: str='.',
                   dist_method: str='neglog',
                   bisto: bool=True,
                   norm_method: str='sites') -> dict:
    """
    Determine the order of sequences for a given clustering solution, as returned by cluster_map. The ordering is
    framed as a Traveling Salesman Problem and uses the LKH solver.

    :param contact_map: An instance of ContactMap to cluster.
    :param clustering: The full clustering solution, derived from the supplied contact map.
    :param seed: Random seed.
    :param cl_list: The list of cluster ids to include in plot. If none, include all ordered clusters.
    :param min_len: Within a cluster exclude sequences that are too short (bp).
    :param min_sig: Within a cluster exclude sequences with weak signal (counts).
    :param max_fold: Within a cluster, exclude sequences that appear to be overly represented.
    :param min_size: Skip clusters which contain too few sequences.
    :param min_extent: Skip clusters whose total extent (bp) is too short.
    :param work_dir: Working directory.
    :param dist_method: Method to use in transforming the contact map to a distance matrix.
    :param bisto: Perform bistochastic matrix balancing.
    :param norm_method: Normalisation method to apply to contact map.
    :return: Map of cluster orders, by cluster id.
    """
    assert os.path.exists(work_dir), 'supplied output path [{}] does not exist'.format(work_dir)

    logger.info('Determining order and orientation')

    if min_extent is None:
        min_extent = contact_map.min_extent
    if min_size is None:
        min_size = contact_map.min_size

    if contact_map.processed_map is None:
        contact_map.set_primary_acceptance_mask(min_len, min_sig, max_fold=max_fold, update=True)
        contact_map.prepare_seq_map(norm=True, bisto=bisto, mean_type='geometric', norm_method=norm_method)

    # analyze all if no subset list was provided
    if cl_list is None:
        cl_list = clustering.keys()
        logger.info('Ordering all suitable clusters')
    else:
        logger.info('Ordering the following specified clusters: {}'.format(np.asarray(cl_list)+1))

    for cl_id in cl_list:

        cl_info = clustering[cl_id]

        cl_size = len(cl_info['seq_ids'])

        if cl_info['extent'] < min_extent:
            logger.debug('Excluding {} too little extent: {} bp'.format(cl_info['name'], cl_info['extent']))
            continue
        elif cl_size < min_size:
            logger.debug('Excluding {} too few sequences: {} '.format(cl_info['name'], cl_size))
            continue

        logger.info('Ordering {} extent: {} size: {}'.format(cl_info['name'], cl_info['extent'], cl_size))

        try:
            # we'll consider only sequences in the cluster
            _mask = np.zeros_like(contact_map.order.mask_vector())
            _mask[cl_info['seq_ids']] = True

            _map = contact_map.get_subspace(external_mask=_mask)

            logger.debug('Cluster size: {} ordering map size: {}'.format(cl_size, _map.shape))

            _ord = contact_map.find_order(_map, work_dir=work_dir, inverse_method=dist_method, seed=seed)

            clustering[cl_id]['order'] = _ord

        except NoneAcceptedException as e:
            logger.warning('{} : cluster {} will be masked'.format(e, cl_info['name']))
            continue
        except TooFewException as e:
            logger.warning('{} : ordering not possible for cluster {}'.format(e, cl_info['name']))

    return clustering
