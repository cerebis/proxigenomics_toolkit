import gzip
import logging
import pickle

import numpy as np
import pandas as pd
import umap
from plotnine import aes, geom_point, ggplot, labs, theme
from sklearn.preprocessing import normalize

from ..io_utils import load_object

logger = logging.getLogger(__name__)


def center_of_mass(embeds: pd.DataFrame) -> np.ndarray:
    """
    Given the sequences involved in a cluster, calculate the center of mass
    embedding vector (weighted by sequence length).
    :param embeds:
    :return: Normalised CoM of cluster
    """
    if len(embeds) == 1:
        return (embeds.values[:,:768]).flatten()
    vals = embeds.iloc[:, :768].values
    lengths = embeds.loc[:, ['length']].values
    return ((lengths * vals).sum(axis=0) / lengths.sum()).flatten()


class MetagenomeEmbeddings(object):
    """
    Determination of cluster embedding vectors as calculated from embedding vectors of
    member sequences. Visualise cluster and sequence vectors as a 2D projection. 

    :ivar embeddings_file: Path to the embeddings file in gzip-pickled format that contains the
        sequence embeddings.
    :type embeddings_file: str
    :ivar clustering_file: Path to the clustering metadata file in a pickled format.
    :type clustering_file: str
    :ivar faidx_file: Path to the sequence index file containing sequence lengths in a tab-separated
        format.
    :type faidx_file: str
    :ivar chunk_names: A list of chunk names extracted from the embeddings file.
    :type chunk_names: list or None
    :ivar chunk_embeds: DataFrame with normalized embedding vectors per chunk.
    :type chunk_embeds: pandas.DataFrame or None
    :ivar seq_names: A list of sequence names extracted from the embeddings file.
    :type seq_names: list or None
    :ivar seq_embeds: DataFrame containing normalized embedding vectors for sequences, and cluster IDs
        where applicable.
    :type seq_embeds: pandas.DataFrame or None
    :ivar cluster_embeds: DataFrame storing the normalized "center of mass" embeddings of clusters.
    :type cluster_embeds: pandas.DataFrame or None
    """

    def __init__(self, embeddings_file: str, clustering_file: str, faidx_file: str) -> None:
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
        
    def load_embeddings(self) -> None:
        """
        Loads and processes embeddings from a pickled dictionary.

        This method reads a gzipped pickle file specified by `self.embeddings_file`.
        The file should contain a dictionary where keys are sequence identifiers
        and values are NumPy arrays of embeddings for the chunks within each
        sequence.

        The function computes two distinct sets of embeddings from this data:
        1.  Chunk Embeddings (`self.chunk_embeds`): A DataFrame containing the
            L2-normalized embedding for each individual chunk. A 'seq' column
            is included to map each chunk back to its parent sequence.
        2.  Sequence Embeddings (`self.seq_embeds`): A DataFrame containing a
            single, L2-normalized embedding for each sequence. This is calculated
            by taking the mean of all chunk embeddings belonging to that
            sequence. The DataFrame is indexed by sequence ID.

        The number of loaded chunk and sequence embeddings is logged to the
        console.
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

    def calculate_cluster_embeddings(self) -> None:
        """
        Calculates a center-of-mass (CoM) embedding for each cluster.

        This method leverages the pre-computed sequence embeddings to generate a
        representative embedding for each cluster. The CoM is calculated as the
        weighted average of the embeddings of all sequences within a cluster,
        where each sequence's contribution is weighted by its length.

        Sequence lengths are read from the `faidx_file`, and the cluster
        memberships are determined from the `clustering_file`.

        This method updates the instance's state in two ways:
        1.  It populates `self.cluster_embeds` with a DataFrame where each row
            contains a cluster ID and its corresponding L2-normalized CoM
            embedding vector.
        2.  It updates `self.seq_embeds` by adding a 'cluster' column,
            assigning a cluster ID to each sequence that belongs to a cluster.
        """
        # We need a source of all sequence lengths, as the clustering
        #   solution can be missing references to some sequences.
        # This is a byproduct of sequences being included in a clustering
        #   result that would not participants in Hi-C.
        fai = pd.read_csv(self.faidx_file, sep='\t', header=None) \
            .drop(columns=range(2, 5)) \
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

    def plot_scatter_projection(self,
                                output_path: str,
                                max_clusters: int=10,
                                verbose: bool=False,
                                metric: str='manhattan',
                                n_components: int=2,
                                n_epochs: int=500,
                                min_dist: float=0.2) -> None:
        """
        Plot the UMAP projection (default 2 components) of the embeddings, with the cluster centers overlaid.

        :param output_path: The file path to save images (extension dictates format).
        :param max_clusters: Number of clusters to display.
        :param verbose: Verbosity of UMAP call.
        :param metric: Distance metric used in projection.
        :param n_components: Number of components in the projection (only first 2 are plotted).
        :param n_epochs: UMAP epochs.
        :param min_dist: UMAP minimum distance.
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