# /bin/python
import logging
import sys

FORMAT = '%(asctime)-15s - %(levelname)s - %(module)10s:%(lineno)-5d - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)
LOG = logging.getLogger(__name__)

help_message = '''
usage: validate.py [-h] [-r ROUTE] [-c COST]

Validate the wire routing output and cost matrix

optional arguments:
  -h, --help            show this help message and exit
  -r ROUTE              Wire Routes for each wire
  -c COST               Cost Array
'''

def parse_args():
    args = sys.argv
    if '-h' in args or '--help' in args:
        print help_message
        sys.exit(1)
    if '-r' not in args or '-c' not in args:
        print help_message
        sys.exit(1)
    parsed = {}
    parsed['route'] = args[args.index('-r') + 1]
    parsed['cost'] = args[args.index('-c') + 1]
    return parsed


def main(args):
    val = validate(args)
    if val:
        LOG.info('Validate succeeded.')
    else:
        LOG.info('Validation failed.')


def validate(args):
    route = open(args['route'], 'r')
    # calculate cost matrix
    lines = route.readlines()
    if len(lines) < 3:
        LOG.error('''Route file contains has less than 3 lines,
        please check for the output format of route file in the handout''')
        return False
    dim = lines[0].split()
    m, n = int(dim[0]), int(dim[1])
    delta = int(lines[1])
    wires = int(lines[2])
    # LOG.info('rows({}), cols({}), wires({})'.format(m, n, wires))
    if len(lines) != wires + 3:
        LOG.error('Route : Expected # of wires %d, Actual # of wires %d'.
                  format(wires, len(lines) - 3))
        return False
    cost_array = [[0] * n for _ in range(m)]
    for i in range(3, len(lines)):
        wire = lines[i]
        path = map(int, wire.split())
        if len(path) % 2 != 0:
            LOG.error('Route: end points doesn\'t come in pairs in line %d'.
                      format(i))
            return False
        points = [(path[2 * i], path[2 * i + 1]) for i in range(len(path) / 2)]
        total_length = 0
        shortest_length = 0
        shortest_length += abs(points[-1][0] - points[0][0])
        shortest_length += abs(points[-1][1] - points[0][1])
        for j in range(len(points) - 1):
            total_length += abs(points[j + 1][0] - points[j][0])
            total_length += abs(points[j + 1][1] - points[j][1])
            add_cost(cost_array, points[j], points[j + 1])
            # remove the cost of the bending end points
            if j + 1 != len(points) - 1:
                cost_array[points[j + 1][1]][points[j + 1][0]] -= 1
        if total_length > delta + shortest_length:
            LOG.error('Route: path length exceeds constrain in line %d'.format(i))
            return False
    LOG.info('Cost Array constructed')
    cost = open(args['cost'], 'r')
    lines = cost.readlines()
    dim = lines[0].split()
    mc, nc = int(dim[0]), int(dim[1])
    if m != mc or n != nc:
        LOG.error('Cost Array: dimension mismatch.')
        return False
    # check # rows
    if mc + 1 != len(lines):
        LOG.error('Cost Array: Incorrect # of rows.')
        return False
    for i in range(1, len(lines)):
        line = map(int, lines[i].split())
        # check # cols
        if len(line) != nc:
            LOG.error('Cost Array: Incorrect # of cols.')
            return False
        # check value
        for j in range(len(line)):
            if cost_array[i - 1][j] != line[j]:
                LOG.error('Cost Array: Value mismatch at (%d, %d)' % (
                    i - 1, j))
                return False
    return True


def add_cost(cost_array, p1, p2):
    y1, x1 = p1[0], p1[1]
    y2, x2 = p2[0], p2[1]
    start_x = x1
    end_x = x2 + 1 if x1 <= x2 else x2 - 1
    step_x = 1 if x1 <= x2 else -1

    start_y = y1
    end_y = y2 + 1 if y1 <= y2 else y2 - 1
    step_y = 1 if y1 <= y2 else -1

    for i in range(start_x, end_x, step_x):
        for j in range(start_y, end_y, step_y):
            cost_array[i][j] += 1


if __name__ == '__main__':
    main(parse_args())
