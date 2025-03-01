from ..io_utils import load_object

import pickle
import gzip
import pandas as pd
import numpy as np
import umap
import logging

from sklearn.preprocessing import normalize
from plotnine import ggplot, geom_point, theme, labs, aes

logger = logging.getLogger(__name__)


def center_of_mass(embeds):
    """
    Given the sequences involved in a cluster, calculate the center of mass
    embedding vector (weighted by sequence length).
    :param embeds:
    :return: normalised CoM of cluster
    """
    if len(embeds) == 1:
        return (embeds.values[:,:768]).flatten()
    v = embeds.iloc[:, :768].values
    l = embeds.loc[:, ['length']].values
    return ((l * v).sum(axis=0) / l.sum()).flatten()


class MetagenomeEmbeddings(object):

    def __init__(self, embeddings_file, clustering_file, faidx_file):
        self.embeddings_file = embeddings_file
        self.clustering_file = clustering_file
        self.faidx_file = faidx_file
        self.chunk_names = None
        self.chunk_embeds = None
        self.seq_names = None
        self.seq_embeds = None
        self.cluster_embeds = None

        self.load_embeddings()
        self.calculate_cluster_embeddings()

    def load_embeddings(self):
        """
        Load the embeddings dictionary produced by the tool seq_embed. Calculate
        the average normalised embedding for each sequence.
        """
        with gzip.open(self.embeddings_file, 'rb') as in_h:
            embeds_dict = pickle.load(in_h)

        chunk_names = []
        seq_names = []
        chunk_embeds = []
        seq_embeds = []

        for k, v in embeds_dict.items():
            chunk_names.extend([k] * v.shape[0])
            chunk_embeds.extend([v])

            seq_names.append(k)
            if v.shape[0] == 1:
                seq_embeds.append(v[0].reshape(1, -1))
            else:
                seq_embeds.append(v.mean(axis=0).reshape(1, -1))

        chunk_embeds = pd.DataFrame(chunk_names, columns=['seq']).join(pd.DataFrame(
            normalize(np.concatenate(chunk_embeds, axis=0), norm='l2')))

        seq_embeds = pd.DataFrame({'seq': seq_names}).join(pd.DataFrame(
            normalize(np.concatenate(seq_embeds, axis=0), norm='l2'))).set_index('seq')

        logger.info(f'Number of chunk embeddings: {len(chunk_embeds)}')
        logger.info(f'Number of sequence embeddings: {len(seq_embeds)}')

        self.chunk_embeds = chunk_embeds
        self.seq_embeds = seq_embeds

    def calculate_cluster_embeddings(self):
        """
        Using the sequence embeddings, calculate a Center of Mass embedding vector for each cluster.
        While performing this calculation, assign the cluster ID to any sequence involved in
        a cluster.
        """
        # We need a source of all sequence lengths, as the clustering
        #   solution can be missing references to some sequences.
        # This is a byproduct of sequences being included in a clustering
        #   result that would not participants in Hi-C.
        fai = pd.read_csv(self.faidx_file, sep='\t', header=None) \
            .drop(columns=range(2,5)) \
            .rename(columns={0: 'seq', 1: 'length'}) \
            .set_index('seq')

        _df = self.seq_embeds.join(fai, how='inner', validate='one_to_one')
        self.seq_embeds['cluster'] = None

        clustering = load_object(self.clustering_file)

        # Calculate mean embedding per cluster.
        cluster_embeds = []
        for cl_id, cl_info in clustering.items():
            cluster_embeds.append(center_of_mass(_df.loc[cl_info['seq_names']]))
            self.seq_embeds.loc[cl_info['seq_names'], 'cluster'] = cl_id

        cluster_embeds = pd.DataFrame(clustering.keys(), columns=['cluster']) \
            .join(pd.DataFrame(
            normalize(np.array(cluster_embeds), norm='l2')), how='inner', validate='1:1')

        logger.info(f'Number of cluster CoM embeddings: {len(cluster_embeds)}')
        self.cluster_embeds = cluster_embeds

    def plot_scatter_projection(self, output_path, max_clusters=10, verbose=False,
                                metric='manhattan', n_components=2, n_epochs=500, min_dist=0.2):
        """
        Plot the UMAP projection (default 2 component) of the embeddings, with the cluster centers overlaid.

        :param output_path: file path to save images (extension dictates format)
        :param max_clusters: number of clusters to display
        :param verbose: verbosity of UMAP call
        :param metric: distance metric used in projection
        :param n_components: number of components in project (only first 2 are plotted)
        :param n_epochs: UMAP epochs
        :param min_dist: UMAP minimum distance
        """

        # prepare a model using the chunked embeddings
        model = umap.UMAP(verbose=verbose, metric=metric, n_epochs=n_epochs,
                          n_components=n_components, min_dist=min_dist)
        model.fit(self.chunk_embeds.loc[:, range(768)])

        # apply the transformation
        chunk_2d = model.transform(self.chunk_embeds.loc[:, range(768)])
        seq_2d = model.transform(self.seq_embeds.loc[:, range(768)])
        cl_2d = model.transform(self.cluster_embeds.loc[:, range(768)])

        # prepare a table linking chunk index to cluster assignment
        _df = self.seq_embeds[['cluster']].join(self.chunk_embeds.set_index('seq')).reset_index()[['cluster']]
        # keep the first N clusters -- bin3C orders clusters largest (extent) to smallest
        accepted_cl = _df.query('cluster < @max_clusters')
        accepted_ix = accepted_cl.index

        p = (ggplot()
             + geom_point(aes(x=chunk_2d[accepted_ix, 0], y=chunk_2d[accepted_ix, 1],
                              colour='factor(accepted_cl["cluster"]+1)'), size=0.5, alpha=0.67)
             + geom_point(aes(x=cl_2d[:max_clusters, 0],
                              y=cl_2d[:max_clusters, 1]), fill='red', color='white', size=4, alpha=1)
             + theme(figure_size=[12,9], aspect_ratio=1)
             + labs(x='dim1', y='dim2', colour='Cluster ID'))

        p.save(filename=output_path, verbose=False)
