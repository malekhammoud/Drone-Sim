# test_tether.py
from tether_control import query_tether

result = query_tether("Attack reported near 71.995, -94.810! Close off the zone with an 800m radius.")
print(result)