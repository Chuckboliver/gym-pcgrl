import argparse

prob_cond_metrics = {
        'binary': ['regions'],
#       'binary': ['path-length'],
#       'binary': ['regions', 'path-length'],
        'zelda': ['num_enemies'],
        'sokoban': ['num_boxes'],
        }

def get_args():
    opts = argparse.ArgumentParser(description='Conditional PCGRL')
    opts.add_argument(
        '-p',
        '--problem',
        help='which problem (i.e. game) to generate levels for (binary, sokoban, zelda, mario, ... rct, simcity???)',
        default='binary')
    opts.add_argument(
        '-r',
        '--representation',
        help='Which representation to use (narrow, turtle, wide, ... cellular-automaton???)',
        default='turtle')
    opts.add_argument(
        '-c',
        '--conditionals',
        nargs='+',
        help='Which game level metrics to use as conditionals for the generator',
        default=prob_cond_metrics['binary'])
    opts.add_argument(
        '--resume',
        help='Are we resuming from a saved training run?',
        action="store_true",)
    opts.add_argument(
        '--experiment_id',
        help='An experiment ID for tracking different runs of experiments with identical hyperparameters.',
        default=None)
    opts.add_argument(
        '--midep_trgs',
        help='Do we sample new (random) targets mid-episode, or nah?',
        action='store_true',)
    opts.add_argument(
        '--load_best',
        help='Whether to load the best saved model of a given run rather than the latest.',
        action='store_true',)
    opts.add_argument(
        '--change_percentage',
        help='Whether to limit the number of changes to the map per episode.'
        action='store_true',)
    opts.add_argument(
        '--observe_sign',
        help='Whether to observe only the sign of the direction of change to reach a target.'
        action='store_true',)
    opts = opts.parse_args()

    return opts
