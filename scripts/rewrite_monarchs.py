import ast
import astor


class AttributeReplacer(ast.NodeTransformer):


    def visit_Attribute(self, node):
        # Replace `cell.attr` with `cell['attr']`
        if isinstance(node.value, ast.Name) and node.value.id == "cell":
            return ast.Subscript(
                value=node.value,
                slice=ast.Index(value=ast.Str(s=node.attr)),
                ctx=node.ctx
            )
        elif isinstance(node.value, ast.Name) and node.value.id == "neighbour_cell":
            return ast.Subscript(
                value=node.value,
                slice=ast.Index(value=ast.Str(s=node.attr)),
                ctx=node.ctx
            )

        # Replace `grid[col][row].attr` with `grid['attr'][col][row]`
        elif isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Subscript) and isinstance(
                node.value.value.value, ast.Name) and node.value.value.value.id == "grid":
            return ast.Subscript(
                value=ast.Subscript(
                    value=ast.Subscript(
                        value=node.value.value.value,
                        slice=ast.Index(value=ast.Str(s=node.attr)),
                        ctx=node.ctx
                    ),
                    slice=node.value.value.slice,
                    ctx=node.ctx
                ),
                slice=node.value.slice,
                ctx=node.ctx
            )
        return self.generic_visit(node)

    def visit_Subscript(self, node):
        # Handle `grid[col][row]` to ensure it becomes `grid['attr'][col][row]`
        if isinstance(node.value, ast.Subscript) and isinstance(node.value.value, ast.Name) and node.value.value.id == "grid":
            # Dynamically extract the attribute name if available
            attr_name = getattr(node, "attr", None)
            if attr_name:
                return ast.Subscript(
                    value=ast.Subscript(
                        value=node.value.value,
                        slice=ast.Index(value=ast.Str(s=attr_name)),
                        ctx=node.ctx
                    ),
                    slice=node.slice,
                    ctx=node.ctx
                )
        return self.generic_visit(node)

def process_file(file_path):
    with open(file_path, "r") as file:
        code = file.read()

    # Parse the code into an AST
    tree = ast.parse(code)

    # Transform the AST
    transformer = AttributeReplacer()
    transformed_tree = transformer.visit(tree)

    # Write the modified code back to the file
    with open(file_path, "w") as file:
        file.write(astor.to_source(transformed_tree))

# Example usage
if __name__ == "__main__":
    # Replace 'your_file.py' with the path to your Python file
    import glob
    import os
    # Get all Python files in the current directory
    python_files = glob.glob("../src/monarchs/**/*.py", recursive=True)
    for file in python_files:
        print('Processing file:', file)
        process_file(file)