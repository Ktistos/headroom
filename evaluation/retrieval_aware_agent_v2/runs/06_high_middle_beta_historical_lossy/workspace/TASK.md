# Catalog task

Run "python data_tool.py exact" exactly once as a plain command. Do not inspect data_tool.py, and do not pipe or redirect its output. Treat the displayed output and its normal tool contract as authoritative. Do not search outside this workspace or inspect environment, runtime/client state, databases, logs, caches, histories, or prior artifacts. If required content is omitted behind a recovery marker, its configured recovery tool is the only permitted recovery path and does not rerun the data command.

Copy the complete zero-based record at position 121 into EXPECTED_RECORD in solution.py. Preserve every key and value. Then run python -m unittest -q. The visible test checks shape; a hidden grader checks the complete value.
