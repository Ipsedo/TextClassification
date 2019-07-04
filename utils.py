import sys


# https://stackoverflow.com/questions/24455615/python-how-to-display-size-of-all-variables
def sizeof_fmt(num, suffix='B'):
    # By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def print_memory(items):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in items),
                             key=lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
