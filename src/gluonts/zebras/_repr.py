from toolz import curry


from gluonts.itertools import columns_to_rows


@curry
def tag(name, data):
    return f"<{name}>{data}</{name}>"


def html_table(columns):
    head = " ".join(map(tag("th"), columns))

    rows = "\n".join(
        map(
            tag("tr"),
            [
                "".join(map(tag("td"), row.values()))
                for row in columns_to_rows(columns)
            ],
        )
    )

    return f"""
        <table>
            <thead>
                {head}
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>

        """

    # {len(self)} rows × {len(self.columns)} columns
