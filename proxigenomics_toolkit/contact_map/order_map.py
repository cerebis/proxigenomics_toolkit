from proxigenomics_toolkit.exceptions import *
import logging
import numpy as np
import os

logger = logging.getLogger(__name__)


def order_clusters(contact_map, clustering, seed, min_len=None, min_sig=None, max_fold=None, min_extent=None,
                   min_size=1, work_dir='.', dist_method='neglog'):
    """
    Determine the order of sequences for a given clustering solution, as returned by cluster_map. The ordering is
    framed as a Travelling Salesman Problem and uses the LKH solver.

    :param contact_map: an instance of ContactMap to cluster
    :param clustering: the full clustering solution, derived from the supplied contact map
    :param seed: random seed
    :param min_len: within a cluster exclude sequences that are too short (bp)
    :param min_sig: within a cluster exclude sequences with weak signal (counts)
    :param max_fold: within a cluster, exclude sequences that appear to be overly represented
    :param min_size: skip clusters which containt too few sequences
    :param min_extent: skip clusters whose total extent (bp) is too short
    :param work_dir: working directory
    :param dist_method: method to use in transforming the contact map to a distance matrix
    :return: map of cluster orders, by cluster id
    """
    assert os.path.exists(work_dir), 'supplied output path [{}] does not exist'.format(work_dir)

    logger.info('Determining order and orientation')

    if min_extent is None:
        min_extent = contact_map.min_extent
    if min_size is None:
        min_size = contact_map.min_size

    if contact_map.processed_map is None:
        contact_map.set_primary_acceptance_mask(min_len, min_sig, max_fold=max_fold, update=True)
        contact_map.prepare_seq_map(norm=True, bisto=True, mean_type='geometric')

    for cl_id, cl_info in clustering.iteritems():

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
            logger.warning('{} : cluster {} will be masked'.format(e.message, cl_info['name']))
            continue
        except TooFewException as e:
            logger.warning('{} : ordering not possible for cluster {}'.format(e.message, cl_info['name']))

    return clustering
