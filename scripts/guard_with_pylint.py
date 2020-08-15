import sys

from pylint import lint
import argparse

THRESHOLD = 9

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="lint for sourcecode.")

    parser.add_argument(
        "-c", "--code_path", nargs="+", default=["xt"], help="""code path""",
    )
    args, _ = parser.parse_known_args()
    print("lint with code path: ", args.code_path)

    run = lint.Run([*args.code_path,
                    "--rcfile=scripts/pylint.conf"], do_exit=False)

    score = run.linter.stats['global_note']

    if score < THRESHOLD:
        print("pylint check is failed with ", score, "which should be ", THRESHOLD)
        sys.exit(1)

    print("pylint check passed with score: {}".format(score))
