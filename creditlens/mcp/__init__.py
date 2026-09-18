"""
CreditLens — Model Context Protocol (MCP) Integration Package
"""

def __getattr__(name: str):
    if name == "mcp":
        from creditlens.mcp.server import mcp
        return mcp
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["mcp"]
