import json
import random

rng = random.Random(104729)
statuses = ["ready"] * 8 + ["paused", "warming"]
rows = [
    {
        "index": index,
        "bucket": f"b{rng.randrange(6)}",
        "status": rng.choice(statuses),
        "value": rng.randrange(11),
        "token": f"code-{rng.randrange(14):02d}",
    }
    for index in range(180)
]
print(json.dumps(rows, indent=2))
