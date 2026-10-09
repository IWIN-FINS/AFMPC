"""Low-CPU live position-error window.

The regular Matplotlib QtAgg canvas can busy-spin on the pool desktop.  This
launcher renders the existing plot off-screen with Agg and presents the
result as a PyQt6 image, leaving the MPC trace follower and all calculations
unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

# Select the non-interactive renderer before importing the existing plot
# module.  The Qt window below only displays the rendered image.
os.environ["MPLBACKEND"] = "Agg"

from PyQt6 import QtCore, QtGui, QtWidgets

from .realtime_position_error_plot import (
    LivePositionErrorPlot,
    OverheadTargetVelocitySource,
    JsonlTraceFollower,
    build_parser,
    load_default_reference,
)


class PositionErrorWindow(QtWidgets.QMainWindow):
    def __init__(self, plot: LivePositionErrorPlot, refresh_ms: int) -> None:
        super().__init__()
        self.plot = plot
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setMinimumSize(800, 600)
        self.setCentralWidget(self.label)
        self.setWindowTitle("FineSUB MPC Position Error — Live")
        self.resize(1200, 850)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(max(250, int(refresh_ms)))
        self.refresh()

    def refresh(self) -> None:
        try:
            self.plot.refresh()
            canvas = self.plot.figure.canvas
            canvas.draw()
            width, height = canvas.get_width_height()
            image = QtGui.QImage(
                bytes(canvas.buffer_rgba()),
                width,
                height,
                4 * width,
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            self.label.setPixmap(QtGui.QPixmap.fromImage(image))
        except Exception as error:  # keep the live window usable on a bad bag
            self.plot.status_text.set_text(f"Plot update error: {error}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window_sec <= 0.0 or args.refresh_ms <= 0:
        raise SystemExit("window-sec and refresh-ms must be positive")
    reference = load_default_reference(args.config)
    overhead_source = (
        None
        if args.overhead_bag is None
        else OverheadTargetVelocitySource(Path(args.overhead_bag).resolve())
    )
    follower = JsonlTraceFollower(
        args.trace_jsonl,
        args.log_directory,
        reference,
        overhead_source=overhead_source,
    )
    plot = LivePositionErrorPlot(
        follower,
        window_s=args.window_sec,
        show_target_speed=args.overhead_bag is not None,
    )
    if args.snapshot_only:
        plot.refresh()
        if not args.save_png:
            raise SystemExit("--snapshot-only requires --save-png")
        plot.figure.savefig(args.save_png, dpi=150)
        return 0
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = PositionErrorWindow(plot, args.refresh_ms)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
