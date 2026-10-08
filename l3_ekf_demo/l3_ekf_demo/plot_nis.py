#!/usr/bin/env python3
"""Task 3: plot the NIS against the range it is supposed to sit in.

Slide: 'CARLA Hands-On', Task 3, 'Whether your covariance is accurate. This is
the task that matters.'

Record first, then plot (either form of the arguments works):
    ros2 topic echo --csv /l3/ekf/nis > nis.csv
    ros2 run l3_ekf_demo plot_nis --ros-args -p csv:=nis.csv -p out:=nis.png
    ros2 run l3_ekf_demo plot_nis --csv nis.csv --out nis.png

Reading the plot is the whole exercise:
    85% or more inside the band -> consistent. The reported covariance
                                   means something.
    otherwise, mean ABOVE 2     -> overconfident. Q or R too small, usually Q.
                                   This is the dangerous one.
    otherwise, mean BELOW 2     -> underconfident. Throwing away information.
                                   Wasteful, safe.
2 is the NIS's expected value: the number of values GNSS reports (x and y).
"""

import argparse
import csv
import sys

from l3_ekf_demo.ekf import VERDICT_TEXT, nis_verdict

LO, HI, DOF = 0.051, 7.378, 2          # 95% interval for m = 2


def read(path: str) -> list[float]:
    out = []
    with open(path) as fh:
        for row in csv.reader(fh):
            for cell in reversed(row):
                try:
                    out.append(float(cell))
                    break
                except ValueError:
                    continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="nis.csv")
    ap.add_argument("--out", default="nis.png")
    argv = sys.argv[1:]
    # ROS-style parameters, "--ros-args -p csv:=nis.csv -p out:=nis.png", as the
    # docstring shows. Before this, they were silently ignored and the
    # defaults used, which only looked right when the values were the defaults.
    ros_params = {}
    if "--ros-args" in argv:
        k = argv.index("--ros-args")
        argv, ros_args = argv[:k], argv[k + 1:]
        for flag, value in zip(ros_args, ros_args[1:]):
            if flag == "-p" and ":=" in value:
                key, val = value.split(":=", 1)
                ros_params[key] = val
    args, _ = ap.parse_known_args(argv)
    args.csv = ros_params.get("csv", args.csv)
    args.out = ros_params.get("out", args.out)

    eps = read(args.csv)
    if not eps:
        sys.exit("no NIS values found in %s" % args.csv)

    verdict, mean, inside = nis_verdict(eps, DOF, LO, HI)
    print("%d fixes, mean NIS %.2f (want about %d), %.0f%% inside [%.3f, %.3f]"
          % (len(eps), mean, DOF, inside, LO, HI))
    print("verdict:", VERDICT_TEXT[verdict])

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; numbers above are the result")
        return

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axhspan(LO, HI, color="#4A7A4E", alpha=0.15,
               label="95%% interval for m=%d" % DOF)
    ax.axhline(DOF, color="#4A7A4E", lw=1.2, ls="--", label="expected, about %d" % DOF)
    ax.plot(eps, lw=0.9, color="#2D6CA2", label="NIS")
    ax.set_xlabel("GNSS fix")
    ax.set_ylabel("NIS")
    ax.set_title("Is the covariance honest?")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("wrote", args.out)
