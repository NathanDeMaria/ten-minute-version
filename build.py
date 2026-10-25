"""
The command line: `python build.py <command> ...`, every command in
`ten_minute_version.build` (see it for what each does). There so a package
that depends on this one -- Cassandra, for `live` -- can import them.
"""

from fire import Fire

from ten_minute_version.build import COMMANDS

if __name__ == "__main__":
    Fire(COMMANDS)
