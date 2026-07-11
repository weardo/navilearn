"""Graphviz DOT and plain-JSON renderings of a concept map."""

from __future__ import annotations

from core.concepts import ConceptMap

# A small qualitative palette; topics cycle through it deterministically.
_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


def _escape(label: str) -> str:
    """Escape a label for safe inclusion in a DOT double-quoted string."""

    return (
        str(label)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", " ")
        .replace("\r", " ")
    )


def _topic_colors(cmap: ConceptMap) -> dict[str, str]:
    topics: list[str] = list(cmap.topics)
    for concept in cmap.concepts:
        if concept.topic and concept.topic not in topics:
            topics.append(concept.topic)
    return {
        topic: _PALETTE[i % len(_PALETTE)] for i, topic in enumerate(topics)
    }


def build_dot(cmap: ConceptMap) -> str:
    """Render the concept map as a Graphviz DOT digraph string.

    Nodes are concepts colored by topic; edges are labeled by relation. All
    labels are safely escaped for use with ``st.graphviz_chart``.
    """

    colors = _topic_colors(cmap)
    lines: list[str] = ["digraph ConceptMap {"]
    lines.append("  rankdir=LR;")
    lines.append('  node [style="filled,rounded", shape=box, fontname="Helvetica"];')
    lines.append('  edge [fontname="Helvetica", fontsize=10];')

    known_nodes: set[str] = set()
    for concept in cmap.concepts:
        color = colors.get(concept.topic, "#cccccc")
        name = _escape(concept.name)
        lines.append(
            f'  "{name}" [fillcolor="{color}", tooltip="{_escape(concept.topic)}"];'
        )
        known_nodes.add(concept.name)

    for edge in cmap.edges:
        # Ensure endpoints exist as nodes even if not in the concept list.
        for endpoint in (edge.source, edge.target):
            if endpoint not in known_nodes:
                lines.append(f'  "{_escape(endpoint)}" [fillcolor="#eeeeee"];')
                known_nodes.add(endpoint)
        lines.append(
            f'  "{_escape(edge.source)}" -> "{_escape(edge.target)}" '
            f'[label="{_escape(edge.relation)}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def graph_json(cmap: ConceptMap) -> dict:
    """Return a plain node/edge dict suitable for JSON serialization."""

    nodes = [
        {"id": c.name, "label": c.name, "topic": c.topic} for c in cmap.concepts
    ]
    known = {c.name for c in cmap.concepts}
    for edge in cmap.edges:
        for endpoint in (edge.source, edge.target):
            if endpoint not in known:
                nodes.append({"id": endpoint, "label": endpoint, "topic": ""})
                known.add(endpoint)
    edges = [
        {"source": e.source, "target": e.target, "relation": e.relation}
        for e in cmap.edges
    ]
    return {"nodes": nodes, "edges": edges}
