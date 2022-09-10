from pygments.style import Style
from pygments.token import (
    Token,
    Comment,
    Keyword,
    Name,
    String,
    Error,
    Generic,
    Number,
    Operator,
    Whitespace,
)

from . import colors


# class Light(Style):
#     background_color = "rgba(0, 0, 0, 0.05)"

#     styles = {
#         Comment: f"italic #888",
#         Comment.Preproc: f"noitalic {colors.GREEN}",
#         Keyword: f"bold {colors.COPPER}",
#         Keyword.Pseudo: "nobold",
#         Keyword.Type: "nobold #B00040",
#         # Operator: "#666666",
#         Operator: f"bold {colors.COPPER}",
#         Operator.Word: f"bold {colors.COPPER}",
#         # Names
#         Name.Builtin: colors.BLUE,
#         Name.Function: f"{colors.BLUE}",
#         Name.Class: f"bold {colors.BLUE}",
#         Name.Namespace: f"bold {colors.BLUE}",  # f"bold {colors.BLUE}",
#         # Name.Exception: "bold #CB3F38",
#         Name.Variable: "#19177C",
#         Name.Constant: colors.BLUE,
#         # Name.Label: "#767600",
#         # Name.Entity: "bold #717171",
#         # Name.Attribute: "#687822",
#         # Name.Tag: "bold #008000",
#         # Name.Decorator: "#AA22FF",
#         String: colors.BLUE,
#         String.Doc: "italic",
#         Number: colors.BLUE,
#     }


class Dark(Style):
    background_color = "rgba(0, 0, 0, 0.0)"

    styles = {
        Whitespace: "#bbbbbb",
        Comment: f"italic #888",
        Comment.Preproc: f"noitalic #ccc",
        Keyword: f"bold {colors.SALMON}",
        Keyword.Pseudo: "nobold",
        Keyword.Type: "nobold #B00040",
        # Operator: "#666666",
        Operator: f"bold {colors.SALMON}",
        Operator.Word: f"bold {colors.SALMON}",
        Name.Builtin: colors.YELLOW,
        Name.Function: f"{colors.GREEN}",
        Name.Class: f"bold {colors.GREEN}",
        Name.Namespace: f"bold {colors.GREEN}",
        Name.Variable: colors.YELLOW,
        Name.Constant: colors.RED,
        Name.Label: colors.YELLOW,
        Name.Attribute: colors.YELLOW,
        Name.Tag: f"bold {colors.SALMON}",
        Name.Decorator: colors.SALMON,
        String: colors.SALMON,
        String.Doc: "italic",
        Number: colors.GREEN,
    }
