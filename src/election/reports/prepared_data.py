"""Print saved prepared-data locations and identities without loading table rows."""
import argparse
from pathlib import Path

from election import paths
from election.pipeline.preparation import prepared_locations
from election.reports._common import print_fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=paths.project_root() / 'outputs' / 'latest_prepared.json')
    args = parser.parse_args(argv)
    locations = prepared_locations(args.data_dir)
    print_fields([(key.replace('_', ' ').capitalize(), value)
                  for key, value in locations.items()])


if __name__ == '__main__':
    main()
