import logging
import os

import networkx as nx
import numpy as np
import pandas as pd
import pyvis

logger = logging.getLogger(__name__)


def prune_graph(g: nx.Graph, min_degree: int) -> nx.Graph:
    """
    For the visualisation, we are only interested in true interactions. After pruning spurious
    edges, we must also remove nodes which become isolated.
    :param g: the interaction graph
    :param min_degree: remove nodes with less than this degree (mainly for removing isolates)
    :return: a pruned graph
    """
    # keep only edges deemed intracellular
    g_tmp = g.copy()
    g_tmp.remove_edges_from(((u, v) for u, v, d in g.edges(data=True) if not d["is_intracellular"]))
    g = g_tmp
    # remove isolated nodes
    g_tmp = g.copy()
    g_tmp.remove_nodes_from((u for u, d in g.nodes(data=True) if g.degree(u) < min_degree))
    g = g_tmp
    # repeat this when pruning aggressively as there may still be isolates
    if min_degree > 1:
        g_tmp = g.copy()
        g_tmp.remove_nodes_from((u for u, d in g.nodes(data=True) if g.degree(u) == 0))
        g = g_tmp
    return g


def report_highly_connected(g: nx.Graph, report_file: str) -> None:
    """
    Generates a report for highly connected nodes in a bipartite graph.
    This function prunes the input graph to retain nodes with a connection
    degree of at least two, then identifies nodes of one partition in the
    bipartite graph. The identified nodes and their connected neighbors
    are written to the specified report file.

    :param report_file: Path to the output file where the highly connected
        nodes and their connections will be recorded.
        The file will contain lines with a node and its neighbors separated
        by commas.
    :type report_file: str
    :param g: An input bipartite graph from the NetworkX library. The
        graph should have a "bipartite" attribute in node data to identify
        partitions.
    :type g: nx.Graph
    :return: This function does not return a value, as it operates by
        writing to a file and logging relevant information.
    :rtype: None
    """
    g = prune_graph(g, 2)
    seq_nodes = sorted({n for n, d in g.nodes(data=True) if d["bipartite"] == 0})
    with open(report_file, "w") as h_output:
        h_output.write("sequence,degree,classification,containing_bins\n")
        for u in seq_nodes:
            h_output.write(f'{u},{g.degree(u)},{g.nodes[u]["mge_status"]},\"{" ".join(g.neighbors(u))}\"\n')
    logger.info(f"There were {len(seq_nodes)} sequences with >= 2 bins")


def make_graph_representation(pred_filename: str,
                              mge_filename: str,
                              binqc_filename: str,
                              min_degree: int=1) -> nx.Graph:
    """
    For a given sample, combine the data from prediction with the collated mge report and collated binning QC
    to produce a NetworkX graph. The graph is undirected and bipartite (although in NetworkX bipartite is merely
    an attribute).
    Nodes contain a number of attributes, which can be used downstream to improve visualisations.
    - intrinics: gc, cov, sites, length, uf.
    - mge results: mge, virus, plasmid, chromosome
    - bin taxonomy
    Edges contain features:
    - similarity, linkage_z, cov_z, sites_z, freq_z, class variable: intra_z
    - classifier score: intracellular_score
    - classifier decision: is_intracellular
    - used in training (bool)
    - bin3C edge was deemed "intracluster", i.e: intra_z
    - number of contacts

    :param pred_filename: Filename of the prediction file.
    :param mge_filename: Filename of the mge file.
    :param binqc_filename: Filename of the binning QC file.
    :param min_degree: Minimum degree of nodes to be considered for the graph.
    :return: A NetworkX graph instance.
    """

    # read the tables and join, clean-up some attributes that tools like Gephi stumble over.
    df_plot = (
        pd.read_csv(pred_filename)
        .set_index("seq")
        .join(pd.read_csv(mge_filename).set_index("name"), how="left", validate="m:1", rsuffix="_mge")
        .reset_index()
        .set_index("cluster_name")
        .join(pd.read_csv(binqc_filename, index_col=0, header=[0, 1]).loc[:, ("GTDBtk")], how="left", rsuffix="_gtdb")
        .reset_index()
        .replace("-", "")
    )

    g = nx.Graph()

    for i, row in df_plot.iterrows():

        seq_id = str(row['seq'])
        bin_id = str(row['cluster_name'])

        # add sequence node if new
        if not g.has_node(seq_id):
            g.add_node(
                seq_id,
                bipartite=0,
                length=row["length_u"],
                gc=row["gc_u"],
                cov=row["cov_u"],
                sites=row["sites_u"],
                uf=row["uf_u"],
                mge_status=row["mge_status"],
                # these are blank so that a unified node-table (downstream-tool dependent) has entries for all nodes
                domain="",
                phylum="",
                _class="",
                order="",
                family="",
                genus="",
                species="",
            )

        # add bin node if new
        if not g.has_node(bin_id):
            g.add_node(
                bin_id,
                bipartite=1,
                length=row["length_v"],
                gc=row["gc_v"],
                cov=row["cov_v"],
                sites=row["sites_v"],
                uf=row["uf_v"],
                domain=row["domain_gtdb"],
                phylum=row["phylum_gtdb"],
                _class=row["class_gtdb"],
                order=row["order_gtdb"],
                family=row["family_gtdb"],
                genus=row["genus_gtdb"],
                species=row["species_gtdb"],
                # again, for the sake of unified node tables
                mge_status="",
            )

        # chuck a wobbly if we attempt to make an edge twice
        assert not g.has_edge(seq_id, bin_id), (
            f"Edge ({seq_id},{bin_id}) already exists"
        )

        # add the edge between the sequence and the bin
        g.add_edge(
            seq_id,
            bin_id,
            weight=row["intracellular_score"],
            training=row["train"],
            intra_z=row["intra_z"],
            similarity=row["similarity"],
            freq_z=row["freq_z"],
            cov_z=row["cov_z"],
            sites_z=row["sites_z"],
            linkage_z=row["linkage_z"],
            contacts=row["contacts"],
            intracellular_score=row["intracellular_score"],
            is_intracellular=row["is_intracellular"],
        )

    logger.debug(f'The full graph representation contained {g.order()} nodes and {g.size()} edges.')

    g = prune_graph(g, min_degree=min_degree)
    logger.debug(f'The pruned graph representation contained {g.order()} nodes and {g.size()} edges.')

    return g


# A simple conversion of MGE classification text to an int.
_NODE_STYLING : dict[str, dict] = {
    'chromosome': {'id': 1, 'color': 'green', 'name': 'Chromosome DNA', 'type': 'Contig'},
    'mge': {'id': 2, 'color': 'purple', 'name': 'Extrachromosomal DNA', 'type': 'Contig'},
    'virus': {'id': 3, 'color': 'red', 'name': 'Virus DNA', 'type': 'Contig'},
    'plasmid': {'id': 4, 'color': 'blue', 'name': 'Plasmid DNA', 'type': 'Contig'},
}
_BIN_STYLING : dict = {'id': 10, 'color': 'grey', 'name': 'Genome Bin', 'type': 'Genome Bin'}

def add_pyvis_attributes(g: nx.Graph) -> None:
    """
    Add PyVis-specific attributes to the nodes of a NetworkX graph.

    This function modifies the input graph in place by adding attributes to its nodes
    that are specific to PyVis visualizations. These attributes include group,
    mge_status, title, and size. The role and status of the node within the graph
    are used to determine specific graphical and descriptive settings.

    :param g: A NetworkX graph. The graph should have nodes with attributes "bipartite",
              "mge_status", and "length" pre-defined.
    :type g: nx.Graph
    :return: None
    """
    def node_report(node_attr: dict, node_type: str, node_name: str) -> str:
        """
        Helper function to create an informative tooltip in PyVis. HTML parsing is not currently possible.
        :param node_attr: NetworkX node data dictionary
        :param node_type: node type (e.g. 'Contig')
        :param node_name: node name (e.g. 'Chromosome DNA')
        :return: tooltip string
        """
        ret = (
            f"Type: {node_type}\n"
            f"Length: {node_attr['length']:,} bp\n"
            f"Coverage: {node_attr['cov']}\n"
            f"GC: {node_attr['gc']}\n"
        )

        if node_attr["bipartite"] == 1:
            ret += (
                f"Taxonomy:\n"
                f"d__{node_attr['domain']}:\n"
                f"p__{node_attr['phylum']}:\n"
                f"c__{node_attr['_class']}:\n"
                f"o__{node_attr['order']}:\n"
                f"f__{node_attr['family']}:\n"
                f"g__{node_attr['genus']}"
            )
        else:
            ret += f"Classification: {node_name}"
        return ret

    # Define some additional fields used by PyVis.
    # By default, "group" is used for colouring nodes.
    for u, d in g.nodes(data=True):
        _style = _BIN_STYLING if d['bipartite'] == 1 else _NODE_STYLING[d['mge_status']]
        d['group'] = _style['id']
        # d['color'] = _style['color']
        # this is used for tooltips
        d['title'] = node_report(d, _style['type'], _style['name'])
        # sizing nodes based on their length.
        d['size'] = np.sqrt(0.001 * d['length'])

    for u, v, d in g.edges(data=True):
        u, v = sorted([u, v])
        d['title'] = (
            f"Interaction: {u} \u2194 {v}\n"
            f"Score: {d['intracellular_score']:.2f}\n"
            f"Contacts: {d['contacts']}\n"
            f"Intracluster: {bool(d['intra_z'])}\n"
            f"Training: {d['training']}\n"
            "\n"
            f"Similarity: {d['similarity']:.2f}\n"
            f"Linkage_z: {d['linkage_z']:.2f}\n"
            f"Cov_z: {d['cov_z']:.2f}\n"
            f"Sites_z: {d['sites_z']:.2f}\n"
            f"Freq_z: {d['freq_z']:.2f}\n"
        )

def create_interactive_visualisation(output_dir: str,
                                     prediction_file: str,
                                     mge_summary_file: str,
                                     binqc_summary_file: str,
                                     min_degree: int=1,
                                     enable_menu: bool=False,
                                     enable_sidebar: bool=False) -> None:
    """
    Creates an interactive visualization of an interaction graph and saves it as an HTML
    file. This function processes prediction data, summary files, and minimum degree
    threshold to construct a graph representation, which is then styled and rendered
    using pyvis for interactive visualization.

    :param output_dir: Path to the output directory where the HTML file will be saved
    :type output_dir: str
    :param prediction_file: Path to the prediction file containing interaction data
    :type prediction_file: str
    :param mge_summary_file: Path to the summary file for MGE (mobile genetic elements)
    :type mge_summary_file: str
    :param binqc_summary_file: Path to the summary file for bin quality control
    :type binqc_summary_file: str
    :param min_degree: Minimum degree threshold to filter the interaction nodes
    :type min_degree: int
    :param enable_menu: Whether to enable interactive menus such as select and filter
    :type enable_menu: bool
    :param enable_sidebar: Whether to enable the sidebars which control styling and physics.
    :return: None
    :rtype: None
    """
    interaction_graph = make_graph_representation(prediction_file,
                                                  mge_summary_file,
                                                  binqc_summary_file,
                                                  min_degree)

    # write the interaction to a standard graph format
    nx.write_graphml(interaction_graph,
                     os.path.join(output_dir, f'interactions_DEG{min_degree}.graphml'))

    # write a ragged table of all sequences and their containing bins with degree >= 2.
    report_highly_connected(interaction_graph,
                            os.path.join(output_dir, 'highly_connected.csv'))

    add_pyvis_attributes(interaction_graph)

    network = pyvis.network.Network(
        height='100vh',
        width='100%',
        cdn_resources='remote',
        select_menu=enable_menu,
        filter_menu=enable_menu
    )

    network.from_nx(interaction_graph)

    network.barnes_hut(
        gravity=-2000,
        central_gravity=0.5,
        spring_length=95,
        spring_strength=0.04,
        damping=0.09,
        overlap=0.01,
    )
    network.inherit_edge_colors("both")

    if enable_sidebar:
        # Layout bug results in sidebar always below the viewport. Adjust
        # the height to permit scrolling.
        logger.info('Reducing viewport to h:50%, w:50% to allow for sidebar')
        network.show_buttons(filter_=True)
        network.height = '80vh'
        network.width = '100%'

    if enable_menu and not enable_sidebar:
        # accomodate the menubar
        network.height = "90vh"

    network.write_html(os.path.join(output_dir, f'interactions_DEG{min_degree}.html'), notebook=False)
