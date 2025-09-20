
def register_tool(name=None, schema=None):
    def decorator(func):
        func._mcp_tool_name = name
        func._mcp_schema = schema
        return func
    return decorator
