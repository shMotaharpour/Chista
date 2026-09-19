"""tile_dp: the tile lifecycle graph, and the contractor that prices it.

`graph` is the artifact and its reader, `contractor` the DP over it, `tile_state` the
node key, `chains` the base definition of a chain plus the loader for the built ones. A
package __init__ that imported all four would load the chain registry on any import of
any of them, so it imports nothing: reach for the module you want.
"""
