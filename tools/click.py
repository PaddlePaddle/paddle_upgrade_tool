# ref: https://github.com/pallets/click/
import sys

_ansi_colors = {
    "black": 30,
    "red": 31,
    "green": 32,
    "yellow": 33,
    "blue": 34,
    "magenta": 35,
    "cyan": 36,
    "white": 37,
    "reset": 39,
    "bright_black": 90,
    "bright_red": 91,
    "bright_green": 92,
    "bright_yellow": 93,
    "bright_blue": 94,
    "bright_magenta": 95,
    "bright_cyan": 96,
    "bright_white": 97,
}

_ansi_reset_all = "\033[0m"

def _interpret_color(color, offset=0):
    if isinstance(color, int):
        return "{};5;{}".format(38 + offset, color)

    if isinstance(color, (tuple, list)):
        r, g, b = color
        return "{};2;{};{};{}".format(38 + offset, r, g, b)

    return str(_ansi_colors[color] + offset)

def style(
    text,
    fg=None,
    bg=None,
    bold=None,
    dim=None,
    underline=None,
    blink=None,
    reverse=None,
    reset=True,
):
    if not isinstance(text, str):
        text = str(text)

    if sys.platform.lower() == 'win32':
        return text

    bits = []

    if fg:
        try:
            bits.append("\033[0;{}m".format(_interpret_color(fg)))
        except KeyError:
            raise TypeError("Unknown color {}".format(fg))

    if bg:
        try:
            bits.append("\033[0;{}m".format(_interpret_color(bg, 10)))
        except KeyError:
            raise TypeError("Unknown color {}".format(fg))

    if bold is not None:
        bits.append("\033[{}m".format(1 if bold else 22))
    if dim is not None:
        bits.append("\033[{}m".format(2 if dim else 22))
    if underline is not None:
        bits.append("\033[{}m".format(4 if underline else 24))
    if blink is not None:
        bits.append("\033[{}m".format(5 if blink else 25))
    if reverse is not None:
        bits.append("\033[{}m".format(7 if reverse else 27))
    bits.append(text)
    if reset:
        bits.append(_ansi_reset_all)
    return "".join(bits)

def secho(message=None, nl=True, err=False, **styles):
    if message is not None:
        message = style(message, **styles)

    return echo(message, nl=nl, err=err)

def echo(message=None, nl=True, err=False):
    if err:
        out = sys.stderr
    else:
        out = sys.stdout

    # Convert non bytes/text into the native string type.
    if message is not None and not isinstance(message, str):
        message = str(message)

    if nl:
        message = message or ""
        if isinstance(message, str):
            message += "\n"
        else:
            message += b"\n"

    if message:
        out.write(message)
    out.flush()

class Abort(RuntimeError):
    """An internal signalling exception that signals Click to abort."""

def confirm(text, default=False, abort=False, prompt_suffix=": ", err=False):
    prompt = "{} [{}]{}".format(text, "Y/n", prompt_suffix)
    while 1:
        try:
            # Write the prompt separately so that we get nice
            # coloring through colorama on Windows
            echo(prompt, nl=False, err=err)
            value = input("").lower().strip()
        except (KeyboardInterrupt, EOFError):
            raise Abort()
        if value in ("y", "yes"):
            rv = True
        elif value in ("n", "no"):
            rv = False
        elif value == "":
            rv = default
        else:
            echo("Error: invalid input", err=err)
            continue
        break
    if abort and not rv:
        raise Abort()
    return rv
