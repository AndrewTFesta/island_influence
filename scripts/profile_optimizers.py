"""
@title

@description

"""
import argparse
import cProfile


def build5():
    arr = [a for a in range(0, 1000000) if (a & 1) == 0]
    return


def build4():
    arr = [a for a in range(0, 1000000) if a % 2 == 0]
    return


def build3():
    arr = []
    for a in range(0, 1000000):
        if a % 2 == 0:
            arr.append(a)
    return


def build2():
    arr = []
    for a in range(0, 1000000):
        if check_even(a):
            arr.append(a)
    return


def check_even(x):
    if x % 2 == 0:
        return x
    else:
        return None


def build():
    arr = []
    for a in range(0, 1000000):
        arr.append(a)
    return


def deploy():
    print('Array deployed!')
    return


def test():
    build()
    deploy()
    return


def main(main_args):
    """
    https://likegeeks.com/python-profiling/
    cProfile.run(statement, filename=None, sort=-1)

    ncalls: represents the number of calls.
    tottime: denotes the total time taken by a function. It excludes the time taken by calls made to sub-functions.
    percall: (tottime)/(ncalls)
    cumtime: represents the total time taken by a function as well as the time taken by subfunctions called by the parent function.
    percall: (cumtime)/( primitive calls)
    filename:lineno(function): gives the respective data of every function.

    :param main_args:
    :return:
    """
    # todo  create optimizer function calls to compare their execution bottlenecks
    cProfile.run('test()')
    cProfile.run('build2()')
    cProfile.run('build3()')
    cProfile.run('build4()')
    cProfile.run('build5()')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
