"""Running work on a background thread without losing its errors."""

import threading
from typing import Any, Callable, Optional


class BackgroundTask:
    """Runs a callable on a thread and re-raises its exception on :meth:`join`.

    A bare :class:`threading.Thread` swallows whatever the target raises.  That
    matters here: editing runs for hours alongside optimisation, and if it dies
    the run would carry on training against unedited frames and finish with a
    result that looks plausible but is not the requested edit.
    """

    def __init__(self, target: Callable[..., Any], name: Optional[str] = None, **kwargs):
        """
        Args:
            target: The callable to run.
            name: Thread name, used in tracebacks and ``py-spy`` output.
            **kwargs: Keyword arguments forwarded to ``target``.
        """
        self._target = target
        self._kwargs = kwargs
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def _run(self) -> None:
        try:
            self._target(**self._kwargs)
        except BaseException as error:  # re-raised by join()
            self._error = error

    def start(self) -> "BackgroundTask":
        self._thread.start()
        return self

    @property
    def failed(self) -> bool:
        """Whether the target raised. Only meaningful once the thread has ended."""
        return self._error is not None

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the thread, then re-raise anything it raised."""
        self._thread.join(timeout)
        if self._error is not None:
            raise RuntimeError(
                f"the {self._thread.name} thread failed; the scene was not fully edited"
            ) from self._error
