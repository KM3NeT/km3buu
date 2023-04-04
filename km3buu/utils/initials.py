#!/usr/bin/env python
# coding=utf-8
# Filename: initials.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Initialisation script for different KM3BUU parts

Usage:
    initials.py (--proposal=PROPOSAL_PATH)
    initials.py (-h | --help)

    PROPOSAL    setup crosssection tables of proposal based on the standard settings used in KM3BUU
Options:
    -h --help                 Show this screen.
    --proposal=PROPOSAL_PATH  Do PROPOSAL initialisations and write tables to PP_PATH

"""
import logging
from pathlib import Path

FORMAT = '%(asctime)s %(levelname)s %(filename)s -- %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


def main():
    from docopt import docopt
    args = docopt(__doc__)
    if args['--proposal']:
        from km3buu.config import Config
        tablepath = Path(args['--proposal'])
        tablepath.mkdir(parents=True, exist_ok=True)
        Config().proposal_itp_tables = str(tablepath.absolute())
        logging.info(f"Writing PROPOSAL tables to: {tablepath}")

        from km3buu.propagation import _setup_utility, PROPOSAL_LEPTON_DEFINITIONS, PROPOSAL_TARGET_WATER, PROPOSAL_TARGET_ROCK
        import proposal as pp

        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[13](),
                       PROPOSAL_TARGET_WATER)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[15](),
                       PROPOSAL_TARGET_WATER)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[-13](),
                       PROPOSAL_TARGET_WATER)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[-15](),
                       PROPOSAL_TARGET_WATER)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[13](), PROPOSAL_TARGET_ROCK)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[15](), PROPOSAL_TARGET_ROCK)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[-13](),
                       PROPOSAL_TARGET_ROCK)
        _setup_utility(PROPOSAL_LEPTON_DEFINITIONS[-15](),
                       PROPOSAL_TARGET_ROCK)


if __name__ == '__main__':
    main()
