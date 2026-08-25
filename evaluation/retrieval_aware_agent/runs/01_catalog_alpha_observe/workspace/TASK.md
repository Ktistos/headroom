# Exact catalog task

Implement this task by editing `solution.py`. Run `python data_tool.py` exactly once as a plain command. Do not read or modify `data_tool.py`, and do not pipe or redirect its output: the displayed runtime catalog is the only authoritative source. Do not run a baseline test before that catalog command.

If the displayed output contains a `<<ccr:HASH ...>>` marker, call the available Headroom retrieval tool with that hash; do not guess, infer the generator, or substitute a merely shape-compatible record. The visible test checks shape only, but a hidden grader checks the exact record.

Using zero-based indexing, copy the complete object at position **73** into `EXPECTED_RECORD`. Preserve all five keys and their exact values. Then run `python -m unittest -q`.
