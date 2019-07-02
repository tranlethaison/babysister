""""""
from distutils.util import strtobool


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: {}".format(default))

    while 1:
        choice = input(question + prompt).lower().strip()

        try:
            if default is not None and choice == '':
                return strtobool(default)
            else:
                return strtobool(choice)
        except ValueError as err:
            print(err)
            print("Please respond (case insensitive):\n",
                "\ty, yes, t, true, on or 1 for yes\n",
                "\tn, no, f, false, off or 0 for no")
            continue

