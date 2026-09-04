# OSAF-SORS

This repository contains a partial release of the source code for the paper **From Host-Centric to Semantic-Aware Routing: The OSAF-SORS Paradigm Shift for Preferred Selection of Multi-Computing Instances**.

## Source Code Availability

To help readers understand and validate the core algorithmic workflow presented in the paper, this repository provides the core implementation that can currently be made public. Some implementation details of the complete project are involved in pending patent applications and are undergoing institutional technology-transfer review. They therefore cannot be publicly disclosed until these processes are complete. Subject to the completion of the patent and technology-transfer review, we intend to release additional components in the future.

## Repository Structure

```text
.
|-- main.py           # Public API
|-- service.py        # STag, SLA profiles, and routing policies
|-- routing_types.py  # Node resources, candidate paths, and routing decisions
|-- gat.py            # GraphAttentionLayer and MMGAT
|-- d3qn.py           # D3QN inference network
|-- models.py         # Compatibility exports for neural network components
`-- router.py         # OSAF-SORS Routing Planner
```

## Note

This repository is intended to demonstrate and support research on the algorithmic structure of OSAF-SORS.
