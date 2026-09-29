import logging

from rich.markup import escape


class RichNameFormatter(logging.Formatter):
    """Formatter for use with RichHandler(markup=True).

    Wraps the logger name in a Rich style tag so it gets its own color, while
    escaping the actual log message so any brackets in it (URLs, list reprs like
    ``[q_proj, v_proj]``) are NOT parsed as Rich markup.

    Configured from Hydra's job_logging dictConfig via the ``()`` key, e.g.:

        formatters:
          rich:
            (): log_utils.RichNameFormatter
            name_style: "bold cyan"
    """

    def __init__(self, name_style: str = "bold cyan", **kwargs):
        super().__init__(**kwargs)
        self.name_style = name_style

    def format(self, record: logging.LogRecord) -> str:
        message = escape(record.getMessage())
        name = f"[{self.name_style}]{escape(record.name)}[/] "
        return name + message
