"""Small helpers shared by the nodes."""

from __future__ import annotations


def spin(node) -> None:
    """rclpy.spin, quiet on Ctrl+C.

    Jazzy raises ExternalShutdownException from spin() on SIGINT, and when the
    signal lands while spin() is building its wait set, an RCLError about an
    invalid context. Both mean "shutting down"; anything else is a real error.
    (The same helper as l5_tracking_demo's.)
    """
    import rclpy
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        if node.context.ok():
            raise


def stamp_key(stamp) -> int:
    """A header stamp in whole milliseconds: pairs two images of the same tick."""
    return int(round(stamp.sec * 1000.0 + stamp.nanosec * 1e-6))
