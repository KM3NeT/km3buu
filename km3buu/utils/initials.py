#!/usr/bin/env python
# coding=utf-8
# Filename: initials.py
# Author: Johannes Schumann <jschumann@km3net.de>
"""
Initialisation script for different KM3BUU parts

Usage:
    initials.py (--proposal)
    initials.py (-h | --help)

    PROPOSAL    setup crosssection tables of proposal based on the standard settings used in KM3BUU
Options:
    -h --help   Show this screen.

"""
import logging

FORMAT = '%(asctime)s %(levelname)s %(filename)s -- %(message)s'
logging.basicConfig(format=FORMAT)
logging.getLogger().setLevel(logging.INFO)


def main():
    from docopt import docopt
    args = docopt(__doc__)
    if args['--proposal']:
        from km3buu.config import Config
        tablepath = Config().proposal_itp_tables
        logging.info(f"Writing PROPOSAL tables to: {tablepath}")

        from km3buu.propagation import _setup_utility, PROPOSAL_LEPTON_DEFINITIONS, PROPOSAL_TARGET_WATER, PROPOSAL_TARGET_ROCK
        import proposal as pp

        pp.InterpolationSettings.tables_path = '.'
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
