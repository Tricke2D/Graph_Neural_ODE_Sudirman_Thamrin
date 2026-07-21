"""
Pengambilan road network dari OpenStreetMap via OSMnx.
Sumber: notebooks/01_data_collection_preprocessing.ipynb (Flow 1, cell 1.3-1.4)

Area: radius 2.5 km dari Bundaran HI, mencakup koridor Sudirman-Thamrin.
"""

import networkx as nx
import osmnx as ox

# Koordinat pusat area (Bundaran HI)
CENTER_POINT = (-6.1944, 106.8229)  # (latitude, longitude)
DISTANCE_M = 2500  # radius ~2.5 km, cukup untuk koridor Sudirman-Thamrin


def collect_road_network(center_point=CENTER_POINT, distance=DISTANCE_M,
                          network_type="drive", simplify=True):
    """Ambil graph jaringan jalan dari OpenStreetMap di sekitar titik pusat."""
    G = ox.graph_from_point(
        center_point, dist=distance,
        network_type=network_type, simplify=simplify,
    )
    print(f"Graph berhasil di-download! Jumlah nodes: {G.number_of_nodes()}, "
          f"edges: {G.number_of_edges()}")
    return G


def keep_largest_component(G):
    """Pastikan graph strongly connected -- diperlukan agar propagasi GNN
    antar node valid (tidak ada node yang terisolasi)."""
    n_components = nx.number_strongly_connected_components(G)
    print(f"Jumlah strongly connected components: {n_components}")

    if n_components > 1:
        try:
            G = ox.truncate.largest_component(G, strongly=True)
        except AttributeError:
            G = ox.utils_graph.get_largest_component(G, strongly=True)
        print(f"Setelah filter -> nodes: {G.number_of_nodes()}, "
              f"edges: {G.number_of_edges()}")
    return G


if __name__ == "__main__":
    G = collect_road_network()
    G = keep_largest_component(G)
    ox.save_graphml(G, "sudirman_thamrin_raw.graphml")
