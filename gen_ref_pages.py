"""Generate the code reference pages."""

from pathlib import Path
import os
import mkdocs_gen_files
nav = mkdocs_gen_files.Nav()

for path in sorted(Path("uaiDiffusers").rglob("*.py")):  # 

    if "__init__" in path.name or "__main__" in path.name:  #
        pass
    else:
        module_path = path.relative_to("uaiDiffusers").with_suffix("")  # 
        doc_path = path.relative_to("uaiDiffusers").with_suffix(".md")  # 
        full_doc_path = Path("reference", doc_path)  # 

        parts = list(module_path.parts)

        if parts[-1] == "__init__":  # 
            parts = parts[:-1]
            doc_path = doc_path.with_name("index.md")
            full_doc_path = full_doc_path.with_name("index.md")
        elif parts[-1] == "__main__":
            continue
        print(full_doc_path)
        nav[parts] = doc_path.as_posix()
        with mkdocs_gen_files.open(os.path.abspath(full_doc_path), "w") as fd:  # 
            identifier = ".".join(parts)  # 
            print("::: " + identifier, file=fd)  # 

        mkdocs_gen_files.set_edit_path(full_doc_path, path)  # 
    
with mkdocs_gen_files.open(os.path.abspath("reference/SUMMARY.md"), "w") as nav_file:  # 
    nav_file.writelines(nav.build_literate_nav()) 