import sys
import json
a = [[1,2,3],[1,2]]
a = json.dumps(a)

print(sys.getsizeof(a))
