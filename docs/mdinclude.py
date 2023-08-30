from m2r2 import MdInclude

# This is re-exposing `mdinclunde` from m2r2, without registering `.md`, which
# we want to have handled by myst-parser.


def setup(app):
    app.add_directive("mdinclude", MdInclude)
    app.add_config_value("no_underscore_emphasis", False, "env")
    app.add_config_value("m2r_parse_relative_links", False, "env")
    app.add_config_value("m2r_anonymous_references", False, "env")
    app.add_config_value("m2r_disable_inline_math", False, "env")
    app.add_config_value(
        "m2r_use_mermaid",
        "sphinxcontrib.mermaid" in app.config.extensions,
        "env",
    )

    return {
        "version": "0.0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
